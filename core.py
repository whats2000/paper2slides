# core.py
# This file will contain the core logic for paper2slides,
# refactored from the original CLI scripts to be used by the Streamlit app.

import subprocess
import logging
import yaml
from pathlib import Path
import re
import os
import logging
from openai import OpenAI
import subprocess
import arxiv
from arxiv_to_prompt import process_latex_source
from prompts import PromptManager
from dotenv import load_dotenv
import yaml
import shutil
import threading
from typing import Optional, Tuple
import fitz  # PyMuPDF for PDF text extraction
import hashlib
from PIL import Image
import io
from history import get_history_manager

load_dotenv(override=True)


# Initialize prompt manager
prompt_manager = PromptManager()

def extract_text_from_pdf(pdf_path: str, start_page: int | None = None, end_page: int | None = None) -> str:
    """
    Extract text content from a PDF file using PyMuPDF.
    
    Args:
        pdf_path: Path to the PDF file
        start_page: Starting page number (1-indexed, inclusive). If None, starts from page 1.
        end_page: Ending page number (1-indexed, inclusive). If None, goes to last page.
        
    Returns:
        Extracted text content
    """
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        # Validate and adjust page range (convert from 1-indexed to 0-indexed)
        start_idx = (start_page - 1) if start_page is not None else 0
        end_idx = end_page if end_page is not None else total_pages
        
        # Ensure valid range
        start_idx = max(0, min(start_idx, total_pages - 1))
        end_idx = max(start_idx + 1, min(end_idx, total_pages))
        
        logging.info(f"Extracting text from pages {start_idx + 1} to {end_idx} (out of {total_pages} total pages)")
        
        text_content = []
        for page_num in range(start_idx, end_idx):
            page = doc.load_page(page_num)
            text_content.append(page.get_text())
        
        doc.close()
        return "\n\n".join(text_content)
    except Exception as e:
        logging.error(f"Failed to extract text from PDF: {e}")
        raise


def extract_images_from_pdf(pdf_path: str, output_dir: str, start_page: int | None = None, end_page: int | None = None) -> list[str]:
    """
    Extract images from a PDF file using PyMuPDF and save them to the output directory.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save extracted images
        start_page: Starting page number (1-indexed, inclusive). If None, starts from page 1.
        end_page: Ending page number (1-indexed, inclusive). If None, goes to last page.
        
    Returns:
        List of relative paths to extracted images
    """
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        # Validate and adjust page range (convert from 1-indexed to 0-indexed)
        start_idx = (start_page - 1) if start_page is not None else 0
        end_idx = end_page if end_page is not None else total_pages
        
        # Ensure valid range
        start_idx = max(0, min(start_idx, total_pages - 1))
        end_idx = max(start_idx + 1, min(end_idx, total_pages))
        
        logging.info(f"Extracting images from pages {start_idx + 1} to {end_idx} (out of {total_pages} total pages)")
        
        image_paths = []
        figures_dir = Path(output_dir) / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        image_count = 0
        
        for page_num in range(start_idx, end_idx):
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Skip very small images (likely logos or decorations)
                # Check image dimensions
                try:
                    img_pil = Image.open(io.BytesIO(image_bytes))
                    width, height = img_pil.size
                    
                    # Skip images smaller than 100x100 pixels
                    if width < 100 or height < 100:
                        logging.debug(f"Skipping small image on page {page_num + 1}: {width}x{height}")
                        continue
                    
                    # Skip images with extreme aspect ratios (likely not figures)
                    aspect_ratio = max(width, height) / min(width, height)
                    if aspect_ratio > 10:
                        logging.debug(f"Skipping image with extreme aspect ratio on page {page_num + 1}: {aspect_ratio}")
                        continue
                        
                except Exception as e:
                    logging.warning(f"Could not check image dimensions: {e}")
                
                # Save image with a meaningful name
                image_filename = f"figure_{image_count:03d}.{image_ext}"
                image_path = figures_dir / image_filename
                
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                
                # Store relative path for LaTeX
                relative_path = f"figures/{image_filename}"
                image_paths.append(relative_path)
                image_count += 1
                
                logging.info(f"Extracted image from page {page_num + 1}: {relative_path}")
        
        doc.close()
        
        logging.info(f"Total images extracted: {len(image_paths)}")
        return image_paths
        
    except Exception as e:
        logging.error(f"Failed to extract images from PDF: {e}")
        return []


def generate_pdf_id(pdf_path: str) -> str:
    """
    Generate a unique ID for a PDF file based on its content hash.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        A unique identifier (first 12 chars of SHA256 hash)
    """
    with open(pdf_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    return f"pdf_{file_hash[:12]}"

def search_arxiv(query: str, max_results: int = 3) -> list[arxiv.Result]:
    """
    Searches arXiv for a given query and returns the top results.
    """
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    return list(search.results())


def edit_slides(
    beamer_code: str, 
    instruction: str, 
    api_key: str, 
    model_name: str, 
    base_url: str | None = None,
    paper_id: str = "",
    use_paper_context: bool = True
) -> str | None:
    """
    Edits the Beamer code based on the user's instruction.
    
    Args:
        beamer_code: Current Beamer LaTeX code
        instruction: User's editing instruction
        api_key: API key for LLM
        model_name: Model name to use
        base_url: Optional base URL for API
        paper_id: Paper ID to load latex source from workspace
        use_paper_context: Whether to include original paper source as context (default True)
    """
    # Load latex_source from workspace if requested
    latex_source = ""
    if use_paper_context and paper_id:
        latex_source = load_latex_source(f"source/{paper_id}/")
        if latex_source:
            logging.info(f"Loaded original paper source for editing context (paper {paper_id})")
        else:
            logging.debug(f"No original paper source found for paper {paper_id}")
    
    # Use PromptManager to get prompts from YAML config (interactive_edit stage)
    system_message, user_prompt = prompt_manager.build_prompt(
        stage="interactive_edit",
        beamer_code=beamer_code,
        user_instructions=instruction,
        latex_source=latex_source,
    )

    try:
        # Resolve API key and base_url (supports multiple LLM providers)
        resolved_api_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("DASHSCOPE_API_KEY")
        )
        if not resolved_api_key:
            raise RuntimeError(
                "No API key provided. Set OPENAI_API_KEY or DASHSCOPE_API_KEY."
            )
        client_kwargs = {"api_key": resolved_api_key}
        
        # Determine which provider is being used and set base_url
        if base_url:
            # Use provided base_url (from UI)
            client_kwargs["base_url"] = base_url
        elif resolved_api_key == os.environ.get("DASHSCOPE_API_KEY"):
            # DashScope provider
            client_kwargs["base_url"] = (
                os.environ.get("DASHSCOPE_BASE_URL") 
                or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
        elif os.environ.get("OPENAI_BASE_URL"):
            # Custom OpenAI-compatible provider (DeepSeek, Together, local, etc.)
            client_kwargs["base_url"] = os.environ.get("OPENAI_BASE_URL")

        client = OpenAI(**client_kwargs)
        # Choose model (auto-adjust for DashScope if an OpenAI model is specified)
        model_to_use = model_name
        if (
            isinstance(client_kwargs.get("base_url"), str)
            and "dashscope.aliyuncs.com" in client_kwargs["base_url"]
            and isinstance(model_name, str)
            and (
                model_name.startswith("gpt-")
                or model_name.startswith("o1")
                or model_name.startswith("o3")
            )
        ):
            model_to_use = os.environ.get("DASHSCOPE_MODEL", "qwen-plus")
        response = client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = extract_content_from_response(response)
        if not content:
            return None
            
        sanitized_content = sanitize_frametitles(content)
        
        # If paper_id is provided, try to compile and fix if needed
        if paper_id:
            logging.info("Attempting to compile edited slides...")
            compiled_code = try_compile_with_fixes(
                sanitized_content,
                paper_id,
                api_key,
                model_name,
                base_url,
                max_retries=3,
                use_paper_context=use_paper_context,
            )
            
            if compiled_code:
                logging.info("✓ Edit successful and compiled")
                return compiled_code
            else:
                logging.error("✗ Edit failed to compile after all fix attempts")
                return None
        else:
            # No paper_id, return without compiling
            return sanitized_content
            
    except Exception as e:
        logging.error(f"Error editing slides: {e}")
        # Provide guidance for DashScope access issues
        try:
            if "dashscope.aliyuncs.com" in (client_kwargs.get("base_url") or "") and (
                "403" in str(e) or "access_denied" in str(e)
            ):
                logging.error(
                    "DashScope access denied. Ensure your key has access to the model. "
                    "Set DASHSCOPE_MODEL to a model you can use (e.g., qwen-plus)."
                )
        except Exception:
            pass
        return None


def extract_frames_from_beamer(beamer_code: str) -> list[tuple[int, str, int, int]]:
    """
    Extract all frames from Beamer code.
    
    Returns a list of tuples: (frame_number, frame_content, start_pos, end_pos)
    where frame_content includes the \\begin{frame} and \\end{frame} tags,
    and start_pos/end_pos are character positions in the original string.
    """
    frames = []
    frame_pattern = r'\\begin\{frame\}.*?\\end\{frame\}'
    
    for match in re.finditer(frame_pattern, beamer_code, re.DOTALL):
        frame_content = match.group(0)
        start_pos = match.start()
        end_pos = match.end()
        frame_number = len(frames) + 1
        frames.append((frame_number, frame_content, start_pos, end_pos))
    
    return frames


def get_frame_by_number(beamer_code: str, frame_number: int) -> str | None:
    """
    Extract a specific frame from Beamer code by frame number (1-indexed).
    
    Args:
        beamer_code: Full Beamer LaTeX code
        frame_number: Frame number to extract (1-indexed, matching PDF page numbers)
        
    Returns:
        Frame content (including \\begin{frame} and \\end{frame}) or None if not found
    """
    frames = extract_frames_from_beamer(beamer_code)
    
    for frame_num, frame_content, _, _ in frames:
        if frame_num == frame_number:
            return frame_content
    
    return None


def replace_frame_in_beamer(beamer_code: str, frame_number: int, new_frame_content: str) -> str | None:
    """
    Replace a specific frame in Beamer code with new content.
    The new content can be one or multiple frames (e.g., when splitting a slide).
    
    Args:
        beamer_code: Full Beamer LaTeX code
        frame_number: Frame number to replace (1-indexed)
        new_frame_content: New frame content (should include \\begin{frame} and \\end{frame}).
                          Can contain multiple frames if splitting.
        
    Returns:
        Updated Beamer code with the frame replaced, or None if frame not found
    """
    frames = extract_frames_from_beamer(beamer_code)
    
    for frame_num, _, start_pos, end_pos in frames:
        if frame_num == frame_number:
            # Replace the frame at the specific position
            # new_frame_content can be one or multiple frames
            updated_code = beamer_code[:start_pos] + new_frame_content + beamer_code[end_pos:]
            return updated_code
    
    return None


def edit_single_slide(
    beamer_code: str, 
    frame_number: int,
    instruction: str, 
    api_key: str, 
    model_name: str,
    base_url: str | None = None,
    paper_id: str = "",
    use_paper_context: bool = True
) -> str | None:
    """
    Edits a specific slide/frame in the Beamer code based on the user's instruction.
    The specified frame is edited according to the instruction. If the instruction
    asks to split the frame, multiple frames will be created to replace the original.
    
    Args:
        beamer_code: Full Beamer LaTeX code
        frame_number: Frame number to edit (1-indexed, matching PDF page numbers)
        instruction: User's editing instruction (can include split instructions)
        api_key: API key for LLM
        model_name: Model name to use
        base_url: Optional base URL for API
        paper_id: Paper ID to load latex source from workspace
        use_paper_context: Whether to include original paper source as context (default True)
        
    Returns:
        Updated full Beamer code with the frame edited (or split into multiple frames), or None on error
    """
    # Load latex_source from workspace if requested
    latex_source = ""
    if use_paper_context and paper_id:
        latex_source = load_latex_source(f"source/{paper_id}/")
        if latex_source:
            logging.info(f"Loaded original paper source for editing context (paper {paper_id})")
        else:
            logging.debug(f"No original paper source found for paper {paper_id}")
    
    # Extract the specific frame
    frame_content = get_frame_by_number(beamer_code, frame_number)
    if not frame_content:
        logging.error(f"Frame {frame_number} not found in Beamer code")
        return None
    
    # Use PromptManager to get prompts from YAML config (interactive_edit_single_slide stage)
    system_message, user_prompt = prompt_manager.build_prompt(
        stage="interactive_edit_single_slide",
        beamer_code=beamer_code,
        frame_number=frame_number,
        frame_content=frame_content,
        user_instructions=instruction,
        latex_source=latex_source,
    )

    try:
        # Resolve API key and base_url (same as edit_slides)
        resolved_api_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("DASHSCOPE_API_KEY")
        )
        if not resolved_api_key:
            raise RuntimeError(
                "No API key provided. Set OPENAI_API_KEY or DASHSCOPE_API_KEY."
            )
        client_kwargs = {"api_key": resolved_api_key}
        
        # Determine which provider is being used and set base_url
        if base_url:
            # Use provided base_url (from UI)
            client_kwargs["base_url"] = base_url
        elif resolved_api_key == os.environ.get("DASHSCOPE_API_KEY"):
            # DashScope provider
            client_kwargs["base_url"] = (
                os.environ.get("DASHSCOPE_BASE_URL") 
                or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
        elif os.environ.get("OPENAI_BASE_URL"):
            # Custom OpenAI-compatible provider
            client_kwargs["base_url"] = os.environ.get("OPENAI_BASE_URL")

        client = OpenAI(**client_kwargs)
        
        # Choose model (auto-adjust for DashScope if needed)
        model_to_use = model_name
        if (
            isinstance(client_kwargs.get("base_url"), str)
            and "dashscope.aliyuncs.com" in client_kwargs["base_url"]
            and isinstance(model_name, str)
            and (
                model_name.startswith("gpt-")
                or model_name.startswith("o1")
                or model_name.startswith("o3")
            )
        ):
            model_to_use = os.environ.get("DASHSCOPE_MODEL", "qwen-plus")
            
        response = client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
            ],
        )
        
        edited_frame_content = extract_content_from_response(response)
        if not edited_frame_content:
            logging.error("Failed to extract edited frame from LLM response")
            return None
        
        # Sanitize the edited frame
        edited_frame_content = sanitize_frametitles(edited_frame_content)
        
        # Replace the frame in the full Beamer code
        updated_beamer_code = replace_frame_in_beamer(beamer_code, frame_number, edited_frame_content)
        
        if not updated_beamer_code:
            logging.error(f"Failed to replace frame {frame_number} in Beamer code")
            return None
        
        # If paper_id is provided, try to compile and fix if needed
        if paper_id:
            logging.info("Attempting to compile edited slide...")
            compiled_code = try_compile_with_fixes(
                updated_beamer_code,
                paper_id,
                api_key,
                model_name,
                base_url,
                max_retries=3,
                use_paper_context=use_paper_context,
            )
            
            if compiled_code:
                logging.info("✓ Single slide edit successful and compiled")
                return compiled_code
            else:
                logging.error("✗ Single slide edit failed to compile after all fix attempts")
                return None
        else:
            # No paper_id, return without compiling
            return updated_beamer_code
        
    except Exception as e:
        logging.error(f"Error editing single slide: {e}")
        return None


def try_compile_with_fixes(
    beamer_code: str,
    paper_id: str,
    api_key: str,
    model_name: str,
    base_url: str | None = None,
    max_retries: int = 3,
    use_paper_context: bool = True,
) -> str | None:
    """
    Try to compile beamer code. If it fails, attempt to fix it using the revise stage.
    Retry up to max_retries times. If all attempts fail, return None.
    
    This function:
    1. Saves beamer_code to a temp file
    2. Tries to compile it
    3. If compilation fails, uses revise stage to fix errors
    4. Retries compilation with fixed code
    5. Repeats up to max_retries times
    6. Returns fixed code on success, None on failure
    
    Args:
        beamer_code: Beamer LaTeX code to compile
        paper_id: Paper ID
        api_key: API key for LLM
        model_name: Model name
        base_url: Optional base URL for API
        max_retries: Maximum number of fix attempts (default 3)
        use_paper_context: Whether to include original paper source during fixes (default True)
        
    Returns:
        Successfully compiled beamer code, or None if all attempts failed
    """
    import tempfile
    import shutil
    
    tex_files_directory = f"source/{paper_id}/"
    slides_tex_path = f"{tex_files_directory}slides.tex"
    pdflatex_path = get_pdflatex_path()
    
    # Create temp file for testing
    temp_tex_path = f"{tex_files_directory}slides_temp.tex"
    
    current_code = beamer_code
    
    for attempt in range(max_retries + 1):  # +1 for initial attempt
        # Save current code to temp file
        try:
            with open(temp_tex_path, "w", encoding="utf-8") as f:
                f.write(current_code)
        except Exception as e:
            logging.error(f"Failed to write temp file: {e}")
            return None
        
        # Try to compile the temp file
        logging.info(f"Compilation attempt {attempt + 1}/{max_retries + 1}...")
        
        try:
            # Pre-sanitize frametitles
            sanitized = sanitize_frametitles(current_code)
            if sanitized and sanitized != current_code:
                with open(temp_tex_path, "w", encoding="utf-8") as f:
                    f.write(sanitized)
                current_code = sanitized
        except Exception:
            pass
        
        # Run pdflatex twice on temp file
        command = [pdflatex_path, "-interaction=nonstopmode", "slides_temp.tex"]
        result1 = subprocess.run(command, cwd=tex_files_directory, capture_output=True, text=True)
        result2 = subprocess.run(command, cwd=tex_files_directory, capture_output=True, text=True)
        
        # Check if PDF was created
        temp_pdf_path = f"{tex_files_directory}slides_temp.pdf"
        if result2.returncode == 0 or Path(temp_pdf_path).exists():
            # Compilation succeeded!
            logging.info(f"✓ Compilation succeeded on attempt {attempt + 1}")
            
            # Clean up temp files
            try:
                for ext in [".aux", ".log", ".nav", ".out", ".snm", ".toc", ".pdf", ".fls", ".fdb_latexmk"]:
                    temp_file = f"{tex_files_directory}slides_temp{ext}"
                    if Path(temp_file).exists():
                        Path(temp_file).unlink()
            except Exception:
                pass
            
            return current_code
        
        # Compilation failed
        if attempt < max_retries:
            # Try to fix it
            logging.warning(f"✗ Compilation failed on attempt {attempt + 1}. Attempting to fix...")
            
            # Run chktex linter on temp file
            try:
                subprocess.run(
                    ["chktex", "-o", "linter_temp.log", "slides_temp.tex"],
                    cwd=tex_files_directory,
                    capture_output=True,
                )
                linter_log_path = f"{tex_files_directory}linter_temp.log"
                if Path(linter_log_path).exists():
                    linter_log = read_file(linter_log_path)
                    Path(linter_log_path).unlink()  # Clean up
                else:
                    linter_log = "No linter output available."
            except Exception:
                linter_log = "Linter not available."
            
            # Load context for fix (respecting use_paper_context flag)
            if use_paper_context:
                latex_source = load_latex_source(tex_files_directory)
            else:
                latex_source = ""
            figure_paths = find_image_files(tex_files_directory)
            
            # Use revise stage to fix
            try:
                system_message, user_prompt = prompt_manager.build_prompt(
                    stage=3,  # revise stage
                    latex_source=latex_source,
                    beamer_code=current_code,
                    linter_log=linter_log,
                    figure_paths=figure_paths,
                )
                
                # Call LLM to fix
                resolved_api_key = (
                    api_key
                    or os.environ.get("OPENAI_API_KEY")
                    or os.environ.get("DASHSCOPE_API_KEY")
                )
                if not resolved_api_key:
                    logging.error("No API key available for fixing")
                    break
                    
                client_kwargs = {"api_key": resolved_api_key}
                if base_url:
                    client_kwargs["base_url"] = base_url
                elif resolved_api_key == os.environ.get("DASHSCOPE_API_KEY"):
                    client_kwargs["base_url"] = (
                        os.environ.get("DASHSCOPE_BASE_URL") 
                        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
                    )
                elif os.environ.get("OPENAI_BASE_URL"):
                    client_kwargs["base_url"] = os.environ.get("OPENAI_BASE_URL")
                
                client = OpenAI(**client_kwargs)
                
                model_to_use = model_name
                if (
                    isinstance(client_kwargs.get("base_url"), str)
                    and "dashscope.aliyuncs.com" in client_kwargs["base_url"]
                    and isinstance(model_name, str)
                    and (model_name.startswith("gpt-") or model_name.startswith("o1") or model_name.startswith("o3"))
                ):
                    model_to_use = os.environ.get("DASHSCOPE_MODEL", "qwen-plus")
                
                response = client.chat.completions.create(
                    model=model_to_use,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                
                fixed_code = extract_content_from_response(response)
                if fixed_code:
                    current_code = sanitize_frametitles(fixed_code)
                    logging.info(f"Generated fix for attempt {attempt + 2}")
                else:
                    logging.error("Failed to generate fix")
                    break
                    
            except Exception as e:
                logging.error(f"Error generating fix: {e}")
                break
        else:
            # Max retries reached
            logging.error(f"✗ All {max_retries + 1} compilation attempts failed")
    
    # Clean up temp files
    try:
        for ext in [".tex", ".aux", ".log", ".nav", ".out", ".snm", ".toc", ".pdf", ".fls", ".fdb_latexmk"]:
            temp_file = f"{tex_files_directory}slides_temp{ext}"
            if Path(temp_file).exists():
                Path(temp_file).unlink()
    except Exception:
        pass
    
    return None


def get_pdflatex_path() -> str:
    """
    Load the pdflatex path from the config file.
    """
    config_path = Path(__file__).parent / "prompts" / "config.yaml"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            return config.get("compiler", {}).get("pdflatex_path", "pdflatex")
    except FileNotFoundError:
        logging.warning(
            f"Config file not found at {config_path}. Using default 'pdflatex'."
        )
        return "pdflatex"
    except (yaml.YAMLError, AttributeError) as e:
        logging.warning(f"Error reading config file: {e}. Using default 'pdflatex'.")
        return "pdflatex"


def compile_latex(
    tex_file_path: str, output_directory: str, pdflatex_path: str = "pdflatex", save_history: bool = True
) -> bool:
    """
    Compiles a LaTeX file to PDF using pdflatex.
    Optionally saves version history after successful compilation.
    
    Args:
        tex_file_path: Path to the .tex file
        output_directory: Directory containing the tex file
        pdflatex_path: Path to pdflatex compiler
        save_history: Whether to save to version history after successful compile (default True)
    
    Returns:
        True on success, False on failure.
    """
    try:
        # Pre-sanitize frametitles in slides.tex to avoid '&' errors
        try:
            full_tex_path = Path(output_directory) / tex_file_path
            if full_tex_path.exists():
                original = full_tex_path.read_text(encoding="utf-8", errors="ignore")
                sanitized = sanitize_frametitles(original)
                if sanitized and sanitized != original:
                    full_tex_path.write_text(sanitized, encoding="utf-8")
        except Exception as san_e:
            logging.debug(f"Sanitization skipped due to error: {san_e}")

        command = [pdflatex_path, "-interaction=nonstopmode", tex_file_path]
        # First run
        result1 = subprocess.run(
            command, cwd=output_directory, capture_output=True, text=True
        )
        # Second run to stabilize refs/outlines if needed
        result2 = subprocess.run(
            command, cwd=output_directory, capture_output=True, text=True
        )
        combined_stdout = (result1.stdout or "") + "\n" + (result2.stdout or "")
        combined_stderr = (result1.stderr or "") + "\n" + (result2.stderr or "")

        pdf_path = Path(output_directory) / Path(tex_file_path).with_suffix(".pdf").name
        if result2.returncode != 0:
            logging.error(
                f"Failed to compile the LaTeX file. Check if {pdflatex_path} is installed and the .tex file is correct."
            )
            logging.error(f"pdflatex output:\n{combined_stdout}\n{combined_stderr}")
            # Fallback: consider success if PDF exists
            if pdf_path.exists():
                logging.warning(
                    "pdflatex returned non-zero exit but PDF was produced. Proceeding as success."
                )
                # Save to history on successful compile (only if save_history is True)
                if save_history:
                    _save_compile_history(full_tex_path, output_directory)
                return True
            return False

        if not pdf_path.exists():
            logging.error("pdflatex succeeded but PDF not found.")
            logging.error(f"pdflatex output:\n{combined_stdout}\n{combined_stderr}")
            return False

        logging.info(f"Successfully compiled {tex_file_path} using {pdflatex_path}.")
        
        # Save to history after successful compilation (only if save_history is True)
        if save_history:
            _save_compile_history(full_tex_path, output_directory)
        
        return True
    except FileNotFoundError:
        logging.error(
            f"Failed to find the pdflatex compiler at '{pdflatex_path}'. Please check your config.yaml or system PATH."
        )
        return False


def _save_compile_history(tex_file_path: Path, output_directory: str) -> None:
    """
    Save version history after successful compilation.
    
    Args:
        tex_file_path: Path to the .tex file
        output_directory: Directory containing the tex file (e.g., "source/2302.11553/")
    """
    try:
        # Extract paper_id from output_directory
        paper_id = Path(output_directory).name
        if not paper_id:
            # Try to get from parent path
            paper_id = Path(output_directory).parts[-1] if Path(output_directory).parts else None
        
        if not paper_id:
            logging.debug("Could not determine paper_id for history saving")
            return
        
        # Read the tex content
        tex_content = tex_file_path.read_text(encoding="utf-8", errors="ignore")
        
        # Save to history
        history = get_history_manager(paper_id)
        history.save_version(tex_content, "Successful compile")
        
    except Exception as e:
        logging.debug(f"Failed to save compile history: {e}")


def read_file(file_path: str) -> str:
    """Read a file and return its contents as a string."""
    # Try different encodings in order of preference
    encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]

    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logging.error(
                f"Error reading file {file_path} with encoding {encoding}: {e}"
            )
            continue

    # If all encodings fail, try reading as binary and decode with errors='replace'
    try:
        with open(file_path, "rb") as file:
            content = file.read()
            return content.decode("utf-8", errors="replace")
    except Exception as e:
        logging.error(f"Failed to read file {file_path} with any encoding: {e}")
        raise


def find_image_files(directory: str) -> list[str]:
    """
    Searches for image files (.pdf, .png, .jpeg, .jpg) in the specified directory and
    returns their paths relative to the specified directory.
    """
    image_extensions = [".pdf", ".png", ".jpeg", ".jpg"]
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in image_extensions):
                relative_path = os.path.relpath(os.path.join(root, file), directory)
                image_files.append(relative_path)
    return image_files


def copy_image_assets_from_cache(arxiv_id: str, cache_dir: str, dest_dir: str) -> None:
    """
    Copy all image assets from arxiv-to-prompt cache into the destination directory,
    preserving relative paths. This ensures includegraphics paths like 'figures/...' resolve
    during compilation.

    Expected cache layout example:
    cache/<arxiv_id>/<arxiv_id>/(figures/... | images/...)
    """
    paper_cache_root = Path(cache_dir) / arxiv_id
    if not paper_cache_root.exists():
        return

    image_extensions = {".pdf", ".png", ".jpeg", ".jpg"}
    for root, _, files in os.walk(paper_cache_root):
        for file in files:
            if any(file.endswith(ext) for ext in image_extensions):
                abs_path = Path(root) / file
                rel_path = abs_path.relative_to(paper_cache_root)
                dest_path = Path(dest_dir) / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(abs_path, dest_path)
                except Exception as e:
                    logging.debug(f"Skipped copying asset {abs_path}: {e}")


def extract_content_from_response(
    response: dict, language: str = "latex"
) -> str | None:
    """
    :param response: Response from the language model
    :param language: Language to extract (default is 'latex')
    :return: Extracted content
    """
    pattern = re.compile(rf"```{language}\s*(.*?)```", re.DOTALL)
    match = pattern.search(response.choices[0].message.content)
    content = match.group(1).strip() if match else None
    return content


def _process_latex_source_worker(
    arxiv_id: str, cache_dir: str, result_container: list
) -> None:
    try:
        latex = process_latex_source(
            arxiv_id,
            keep_comments=False,
            remove_appendix_section=True,
            cache_dir=cache_dir,
        )
        result_container.append((True, latex))
    except Exception as e:
        result_container.append((False, e))


def get_latex_from_arxiv_with_timeout(
    arxiv_id: str, cache_dir: str, timeout_seconds: int = 120
) -> Optional[str]:
    """
    Attempt to retrieve LaTeX source from arXiv using arxiv-to-prompt, but
    give up after timeout_seconds to avoid hanging the UI.
    """
    result_container: list[Tuple[bool, object]] = []
    thread = threading.Thread(
        target=_process_latex_source_worker,
        args=(arxiv_id, cache_dir, result_container),
    )
    thread.daemon = True
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        logging.warning("Timed out retrieving LaTeX from arXiv (arxiv-to-prompt).")
        return None
    if not result_container:
        return None
    success, payload = result_container[0]
    if success:
        return payload if isinstance(payload, str) and payload.strip() else None
    else:
        logging.warning(f"arxiv-to-prompt error: {payload}")
        return None


def extract_definitions_and_usepackage_lines(latex_source: str) -> list[str]:
    """
    Extracts definitions and usepackage lines from LaTeX source
    """
    commands = ["\\def", "\\DeclareMathOperator", "\\DeclarePairedDelimiter"]
    packages_to_comment_out = [
        "amsthm",
        "color",
        "hyperref",
        "xcolor",
        "ragged2e",
        "times",
        "graphicx",
        "enumitem",
    ]
    extracted_lines = []

    lines = latex_source.split("\n")
    for line in lines:
        if any(line.strip().startswith(cmd) for cmd in commands):
            extracted_lines.append(line)
        if line.strip().startswith("\\usepackage"):
            # Skip packages that may conflict with Beamer
            if any(pkg in line for pkg in packages_to_comment_out):
                extracted_lines.append("% " + line)
            else:
                extracted_lines.append(line)
    return extracted_lines


def build_additional_tex(defs_and_pkgs: list[str]) -> str:
    """
    Build ADDITIONAL.tex contents from extracted lines.
    """
    header = [
        "% Auto-generated by paper2slides",
        "% This file aggregates definitions and package imports from the paper.",
    ]
    return "\n".join(header + defs_and_pkgs)


def save_additional_tex(contents: str, dest_dir: str) -> None:
    path = Path(dest_dir) / "ADDITIONAL.tex"
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(contents)


def save_latex_source(latex_source: str, dest_dir: str) -> None:
    """
    Save the original LaTeX source to a file for later reference during editing.
    
    Args:
        latex_source: The LaTeX source content
        dest_dir: Directory to save the file (e.g., "source/2302.11553/")
    """
    path = Path(dest_dir) / "ORIGINAL_PAPER.tex"
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(latex_source)
        logging.info(f"Saved original paper LaTeX source to {path}")
    except Exception as e:
        logging.warning(f"Failed to save original LaTeX source: {e}")


def load_latex_source(source_dir: str) -> str:
    """
    Load the original LaTeX source from the saved file.
    
    Args:
        source_dir: Directory containing the saved file (e.g., "source/2302.11553/")
        
    Returns:
        LaTeX source content, or empty string if not available
    """
    path = Path(source_dir) / "ORIGINAL_PAPER.tex"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            logging.warning(f"Failed to load original LaTeX source: {e}")
    return ""


def add_additional_tex(content: str) -> str:
    r"""
    Ensure that \input{ADDITIONAL.tex} is present. If missing, add near the top after documentclass.
    """
    if not content:
        return content
    if "\\input{ADDITIONAL.tex}" in content:
        return content
    # Insert after documentclass line
    pattern = re.compile(
        r"(\\documentclass\[[^\]]*\]\{beamer\}|\\documentclass\{beamer\})"
    )

    def _inserter(m: re.Match) -> str:
        return m.group(1) + "\n\\input{ADDITIONAL.tex}"

    new_content, count = pattern.subn(_inserter, content, count=1)
    if count == 0:
        logging.warning("\\input{ADDITIONAL.tex} is missing. Added manually.")
        return "\\input{ADDITIONAL.tex}\n" + content
    return new_content


def sanitize_frametitles(beamer_code: str) -> str:
    """
    Escapes unescaped ampersands inside \frametitle and its arguments.
    Also sanitizes titles provided via \begin{frame}{...} with optional [options].
    Handles <...>, [...], and {...} arguments with optional whitespace.
    """
    if not beamer_code:
        return ""

    # 1) Sanitize titles in \begin{frame}[opts]{Title}
    def repl_frame(match):
        begin_frame = match.group(1)
        options = match.group(2) or ""
        title = match.group(3)
        sanitized_options = re.sub(r"(?<!\\)&", r"\\&", options)
        sanitized_title = re.sub(r"(?<!\\)&", r"\\&", title)
        return f"{begin_frame}{sanitized_options}{{{sanitized_title}}}"

    pattern_frame = re.compile(r"(\\begin\{frame\})\s*(\[[^\]]*\])?\s*\{([^}]*)\}")
    beamer_code = pattern_frame.sub(repl_frame, beamer_code)

    # 2) Sanitize explicit \frametitle commands
    def repl(match):
        # Groups: 1=\frametitle, 2=<...>, 3=[...], 4={...} content
        command = match.group(1)
        overlay = match.group(2) or ""
        short_title = match.group(3) or ""
        main_title = match.group(4)

        sanitized_overlay = re.sub(r"(?<!\\)&", r"\\&", overlay)
        sanitized_short_title = (
            re.sub(r"(?<!\\)&", r"\\&", short_title) if short_title else ""
        )
        sanitized_main_title = re.sub(r"(?<!\\)&", r"\\&", main_title)

        return f"{command}{sanitized_overlay}{sanitized_short_title}{{{sanitized_main_title}}}"

    pattern = re.compile(
        r"(\\frametitle)\s*(<[^>]*>)?\s*(\[[^\]]*\])?\s*\{(.*?)\}", re.DOTALL
    )

    return pattern.sub(repl, beamer_code)


def process_stage(
    stage: int,
    latex_source: str,
    beamer_code: str,
    linter_log: str,
    figure_paths: list[str],
    slides_tex_path: str,
    api_key: str,
    model_name: str,
    base_url: str | None = None,
):

    system_message, user_prompt = prompt_manager.build_prompt(
        stage=stage,
        latex_source=latex_source,
        beamer_code=beamer_code,
        linter_log=linter_log,
        figure_paths=figure_paths,
    )

    try:
        # Resolve API key and base_url (supports multiple LLM providers)
        resolved_api_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("DASHSCOPE_API_KEY")
        )
        if not resolved_api_key:
            raise RuntimeError(
                "No API key provided. Set OPENAI_API_KEY or DASHSCOPE_API_KEY."
            )
        client_kwargs = {"api_key": resolved_api_key}
        
        # Determine which provider is being used and set base_url
        if base_url:
            # Use provided base_url (from UI)
            client_kwargs["base_url"] = base_url
        elif resolved_api_key == os.environ.get("DASHSCOPE_API_KEY"):
            # DashScope provider
            client_kwargs["base_url"] = (
                os.environ.get("DASHSCOPE_BASE_URL") 
                or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
        elif os.environ.get("OPENAI_BASE_URL"):
            # Custom OpenAI-compatible provider (DeepSeek, Together, local, etc.)
            client_kwargs["base_url"] = os.environ.get("OPENAI_BASE_URL")

        client = OpenAI(**client_kwargs)
        # Choose model (auto-adjust for DashScope if an OpenAI model is specified)
        model_to_use = model_name
        if (
            isinstance(client_kwargs.get("base_url"), str)
            and "dashscope.aliyuncs.com" in client_kwargs["base_url"]
            and isinstance(model_name, str)
            and (
                model_name.startswith("gpt-")
                or model_name.startswith("o1")
                or model_name.startswith("o3")
            )
        ):
            model_to_use = os.environ.get("DASHSCOPE_MODEL", "qwen-plus")
        response = client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
            ],
        )

        logging.info("Received response from LLM.")

    except Exception as e:
        logging.error(f"Error generating prompt for stage {stage}: {e}")
        # Provide guidance for DashScope access issues
        try:
            if "dashscope.aliyuncs.com" in (client_kwargs.get("base_url") or "") and (
                "403" in str(e) or "access_denied" in str(e)
            ):
                logging.error(
                    "DashScope access denied. Ensure your key has access to the model. "
                    "Set DASHSCOPE_MODEL to a model you can use (e.g., qwen-plus)."
                )
        except Exception:
            pass
        return False

    new_beamer_code = extract_content_from_response(response)

    new_beamer_code = sanitize_frametitles(new_beamer_code)

    if not new_beamer_code:
        logging.error("No beamer code found in the response.")
        return False

    with open(slides_tex_path, "w") as file:
        file.write(new_beamer_code)
    logging.info(f"Beamer code saved to {slides_tex_path}")
    return True


def generate_slides(
    arxiv_id: str,
    use_linter: bool,
    use_pdfcrop: bool,
    api_key: str | None = None,
    model_name: str | None = None,
    base_url: str | None = None,
) -> bool:
    # Use DEFAULT_MODEL from environment if model_name is not provided
    if model_name is None:
        model_name = os.getenv("DEFAULT_MODEL", "gpt-4.1-2025-04-14")
    
    # Define paths
    cache_dir = f"cache/{arxiv_id}"
    tex_files_directory = f"source/{arxiv_id}/"
    slides_tex_path = f"{tex_files_directory}slides.tex"

    # Create directories if not exist
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(tex_files_directory, exist_ok=True)

    # Fetch LaTeX source
    logging.info("Fetching LaTeX source from arXiv...")
    latex_source = get_latex_from_arxiv_with_timeout(arxiv_id, cache_dir)
    if latex_source is None:
        logging.error(
            "Failed to retrieve LaTeX source from arXiv within timeout. Aborting generation."
        )
        return False

    # Extract definitions and packages to build ADDITIONAL.tex
    logging.info("Extracting definitions and packages...")
    defs_pkgs = extract_definitions_and_usepackage_lines(latex_source)
    add_tex_contents = build_additional_tex(defs_pkgs)
    save_additional_tex(add_tex_contents, tex_files_directory)

    # Save the original LaTeX source for later reference during editing
    save_latex_source(latex_source, tex_files_directory)

    # Ensure figures and images referenced by the paper are available under source/<id>/
    try:
        copy_image_assets_from_cache(arxiv_id, cache_dir, tex_files_directory)
    except Exception as e:
        logging.debug(f"Copying image assets skipped due to error: {e}")

    # Add \input{ADDITIONAL.tex} if missing
    latex_source = add_additional_tex(latex_source)

    # Find images under source dir to restrict allowed figures
    figure_paths = find_image_files(tex_files_directory)

    # Stage 1: initial generation
    logging.info("Stage 1: generating slides...")
    if not process_stage(
        1,
        latex_source,
        "",
        "",
        figure_paths,
        slides_tex_path,
        api_key or "",
        model_name,
        base_url,
    ):
        return False

    logging.info("Stage 2: refining slides with update prompt...")
    beamer_code = read_file(slides_tex_path)  # read generated beamer code from stage 1
    if not process_stage(
        2,
        latex_source,
        beamer_code,
        "",
        figure_paths,
        slides_tex_path,
        api_key or "",
        model_name,
        base_url,
    ):
        return False

    # Process stage 3 (if linter is used)
    if not use_linter:
        logging.info("Skipping linter stage. Generation complete.")
        return True

    logging.info("Stage 3: running chktex and revising slides...")
    subprocess.run(
        [
            "chktex",
            "-o",
            "linter.log",
            "slides.tex",
        ],
        cwd=tex_files_directory,
    )
    linter_log = read_file(f"{tex_files_directory}linter.log")

    beamer_code = read_file(slides_tex_path)  # read updated beamer code from stage 2
    if not process_stage(
        3,
        latex_source,
        beamer_code,
        linter_log,
        figure_paths,
        slides_tex_path,
        api_key or "",
        model_name,
        base_url,
    ):
        return False

    logging.info("All stages completed successfully.")
    return True


def generate_slides_from_pdf(
    pdf_path: str,
    paper_id: str,
    use_linter: bool,
    use_pdfcrop: bool,
    api_key: str | None = None,
    model_name: str | None = None,
    base_url: str | None = None,
    dashscope_base_url: str | None = None,
    start_page: int | None = None,
    end_page: int | None = None,
) -> bool:
    """
    Generate slides from a local PDF file (not from arXiv).
    
    Args:
        pdf_path: Path to the PDF file
        paper_id: Unique identifier for this paper (will be generated if from uploaded PDF)
        use_linter: Whether to use ChkTeX linter
        use_pdfcrop: Whether to use pdfcrop (not used for direct PDF)
        api_key: OpenAI/DashScope API key
        model_name: Model to use for generation (defaults to DEFAULT_MODEL env var)
        base_url: Base URL for OpenAI-compatible API (overrides env)
        dashscope_base_url: Base URL for DashScope API (overrides env)
        start_page: Starting page number (1-indexed, inclusive). If None, starts from page 1.
        end_page: Ending page number (1-indexed, inclusive). If None, goes to last page.
        
    Returns:
        True if successful, False otherwise
    """
    # Use DEFAULT_MODEL from environment if model_name is not provided
    if model_name is None:
        model_name = os.getenv("DEFAULT_MODEL", "gpt-4.1-2025-04-14")
    # Set base URLs in environment if provided (for process_stage to use)
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
    if dashscope_base_url:
        os.environ["DASHSCOPE_BASE_URL"] = dashscope_base_url
    
    # Define paths
    tex_files_directory = f"source/{paper_id}/"
    slides_tex_path = f"{tex_files_directory}slides.tex"

    # Create directories if not exist
    os.makedirs(tex_files_directory, exist_ok=True)

    # Extract text from PDF
    logging.info(f"Extracting text from PDF: {pdf_path}")
    if start_page or end_page:
        page_range_msg = f" (pages {start_page or 1} to {end_page or 'end'})"
        logging.info(f"Using page range: {page_range_msg}")
    try:
        pdf_text = extract_text_from_pdf(pdf_path, start_page, end_page)
        if not pdf_text.strip():
            logging.error("No text content extracted from PDF. The PDF might be image-based or empty.")
            return False
    except Exception as e:
        logging.error(f"Failed to extract text from PDF: {e}")
        return False

    # Extract images from PDF
    logging.info(f"Extracting images from PDF: {pdf_path}")
    try:
        figure_paths = extract_images_from_pdf(pdf_path, tex_files_directory, start_page, end_page)
        if figure_paths:
            logging.info(f"Successfully extracted {len(figure_paths)} images from PDF")
        else:
            logging.info("No images found in PDF (or all were too small)")
    except Exception as e:
        logging.warning(f"Failed to extract images from PDF: {e}")
        figure_paths = []

    # Copy the original PDF to the source directory for reference
    try:
        dest_pdf = Path(tex_files_directory) / "original_paper.pdf"
        shutil.copy2(pdf_path, dest_pdf)
        logging.info(f"Copied original PDF to {dest_pdf}")
    except Exception as e:
        logging.warning(f"Failed to copy original PDF: {e}")

    # Create a minimal ADDITIONAL.tex (no LaTeX source to extract from)
    add_tex_contents = build_additional_tex([])
    save_additional_tex(add_tex_contents, tex_files_directory)

    # Since we don't have LaTeX source, we'll format the PDF text as the "source"
    # We'll wrap it in a way that makes it clear this is plain text from a PDF
    formatted_source = f"""% This is text extracted from a PDF file (not LaTeX source)
% The following content should be used to create presentation slides

{pdf_text}
"""

    # Save the extracted PDF text as the "original source" for later reference during editing
    save_latex_source(formatted_source, tex_files_directory)

    # Stage 1: initial generation from PDF text
    logging.info("Stage 1: generating slides from PDF text...")
    if not process_stage(
        1,
        formatted_source,
        "",
        "",
        figure_paths,
        slides_tex_path,
        api_key or "",
        model_name,
        base_url,
    ):
        return False

    logging.info("Stage 2: refining slides with update prompt...")
    beamer_code = read_file(slides_tex_path)
    if not process_stage(
        2,
        formatted_source,
        beamer_code,
        "",
        figure_paths,
        slides_tex_path,
        api_key or "",
        model_name,
        base_url,
    ):
        return False

    # Process stage 3 (if linter is used)
    if not use_linter:
        logging.info("Skipping linter stage. Generation complete.")
        return True

    logging.info("Stage 3: running chktex and revising slides...")
    subprocess.run(
        [
            "chktex",
            "-o",
            "linter.log",
            "slides.tex",
        ],
        cwd=tex_files_directory,
    )
    linter_log = read_file(f"{tex_files_directory}linter.log")

    beamer_code = read_file(slides_tex_path)
    if not process_stage(
        3,
        formatted_source,
        beamer_code,
        linter_log,
        figure_paths,
        slides_tex_path,
        api_key or "",
        model_name,
        base_url,
    ):
        return False

    logging.info("All stages completed successfully.")
    return True

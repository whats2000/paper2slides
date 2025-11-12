"""Compiler functions for compiling LaTeX/Beamer files to PDF."""

import subprocess
import logging
import yaml
from pathlib import Path
from latex_utils import sanitize_frametitles
from file_utils import read_file
from history import get_history_manager


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
    # Import here to avoid circular dependency
    from llm_client import call_llm_for_revision
    from latex_utils import load_latex_source
    from file_utils import find_image_files
    from prompts import PromptManager
    
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
                fixed_code = call_llm_for_revision(
                    latex_source=latex_source,
                    beamer_code=current_code,
                    linter_log=linter_log,
                    figure_paths=figure_paths,
                    api_key=api_key,
                    model_name=model_name,
                    base_url=base_url,
                )
                
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

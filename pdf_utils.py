"""PDF utility functions for extracting text and images from PDF files."""

import logging
import hashlib
from pathlib import Path
from PIL import Image
import io
import fitz  # PyMuPDF


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

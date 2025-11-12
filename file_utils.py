"""File utility functions for reading and searching files."""

import os
import logging


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

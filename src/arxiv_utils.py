"""arXiv utility functions for searching and downloading papers."""

import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Optional, Tuple
import arxiv
from arxiv_to_prompt import process_latex_source


def search_arxiv(query: str, max_results: int = 3) -> list[arxiv.Result]:
    """
    Searches arXiv for a given query and returns the top results.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    return list(client.results(search))


def _process_latex_source_worker(
    arxiv_id: str, cache_dir: str, result_container: list
) -> None:
    """Worker function for processing LaTeX source with timeout."""
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

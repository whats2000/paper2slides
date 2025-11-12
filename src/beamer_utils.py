"""Beamer utility functions for manipulating Beamer presentation frames."""

import re


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

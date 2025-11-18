"""Beamer utility functions for manipulating Beamer presentation frames."""

import re


def extract_frames_from_beamer(beamer_code: str) -> list[tuple[int, str, int, int]]:
    """
    Extract all frames from Beamer code.
    
    Returns a list of tuples: (frame_number, frame_content, start_pos, end_pos)
    where frame_content includes the \\begin{frame} and \\end{frame} tags,
    and start_pos/end_pos are character positions in the original string.
    
    Note: If \\maketitle appears outside of a frame environment, it's treated as frame 1.
    """
    frames = []
    
    # Check for \maketitle that's not inside a \begin{frame}...\end{frame}
    # Look for \maketitle followed by \begin{frame} (with possible whitespace/newlines)
    maketitle_pattern = r'\\maketitle\s*(?=\\begin\{frame\})'
    maketitle_match = re.search(maketitle_pattern, beamer_code)
    
    # Also check if \maketitle appears before any frame at all
    first_frame_pattern = r'\\begin\{frame\}'
    first_frame_match = re.search(first_frame_pattern, beamer_code)
    
    if maketitle_match and (not first_frame_match or maketitle_match.start() < first_frame_match.start()):
        # \maketitle exists before the first frame, treat it as frame 1
        maketitle_full_match = re.search(r'\\maketitle', beamer_code)
        if maketitle_full_match:
            start_pos = maketitle_full_match.start()
            end_pos = maketitle_full_match.end()
            frames.append((1, r'\maketitle', start_pos, end_pos))
    
    # Now extract all regular frames
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
        Frame content (including \\begin{frame} and \\end{frame}, or just \\maketitle for frame 1)
        or None if not found
        
    Note: If \\maketitle appears outside of a frame environment, it's treated as frame 1.
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

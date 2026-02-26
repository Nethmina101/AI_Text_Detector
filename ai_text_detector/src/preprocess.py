from __future__ import annotations

import re
from typing import List

_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")


def split_paragraphs(text: str, min_chars: int = 5) -> List[str]:
    """
    Split the document into true paragraphs as cohesive blocks.
    - True paragraphs (separated by \n\n) are chunked individually.
    - Internal line breaks (e.g. line wraps inside a paragraph) are merged.
    """
    if not text:
        return []

    # Standardize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # A double-newline is almost always a new paragraph in raw PDF strings
    raw_blocks = re.split(r'\n\s*\n', text)

    cleaned_blocks: List[str] = []
    
    for block in raw_blocks:
        # Keep the \n formatting exactly as it is so titles stay separated, 
        # but just trim outer whitespace
        block = block.strip()
        
        # Keep everything except purely empty/tiny artifacts
        if len(block) >= min_chars:
            cleaned_blocks.append(block)

    return cleaned_blocks

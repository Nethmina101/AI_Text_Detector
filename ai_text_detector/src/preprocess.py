from __future__ import annotations

import re
from typing import List

_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")


def split_paragraphs(text: str, min_chars: int = 5) -> List[str]:
    """
    Split the document into true paragraphs as cohesive blocks.
    - True paragraphs (separated by double newlines) are individual blocks.
    - Internal single line breaks are PRESERVED inside each block.
    """
    if not text:
        return []

    # Standardize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Double-newline = new paragraph
    raw_blocks = re.split(r'\n\s*\n', text)

    cleaned_blocks: List[str] = []
    for block in raw_blocks:
        block = block.strip()
        if len(block) >= min_chars:
            cleaned_blocks.append(block)

    return cleaned_blocks


def split_into_lines(paragraph: str, min_chars: int = 5) -> List[str]:
    """
    Split a paragraph into individual lines, preserving the original
    document line structure exactly.  Each line is a separate scorable unit.
    Lines that are too short to score are kept as empty strings so we
    can reconstruct the original layout when rendering.
    """
    if not paragraph or not paragraph.strip():
        return []

    lines = paragraph.split("\n")
    return lines

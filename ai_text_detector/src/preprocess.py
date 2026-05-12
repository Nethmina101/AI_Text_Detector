from __future__ import annotations

import re
from typing import List

_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")

# Sentence-ending punctuation followed by optional quote/bracket
_SENT_END = re.compile(r'[.!?]["\')\]]?\s*$')


def _rejoin_soft_breaks(block: str) -> str:
    if not block:
        return block

    # fix hyphenated breaks: "word-\nbreak" → "wordbreak"
    block = _HYPHEN_BREAK.sub(r"\1\2", block)

    lines = block.split("\n")
    if len(lines) <= 1:
        return block

    joined_parts: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped: 
            continue
        joined_parts.append(stripped)

    return " ".join(joined_parts)


def split_paragraphs(text: str, min_chars: int = 5) -> List[str]:
    """
    Split the document into true paragraphs as cohesive blocks.
    - True paragraphs (separated by double newlines) are individual blocks.
    - Internal single line breaks are REJOINED (they are visual PDF wrapping).
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
            # Rejoin soft line breaks within the paragraph PDF-extracted text matches the structure of pasted text
            block = _rejoin_soft_breaks(block)
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

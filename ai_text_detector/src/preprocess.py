from __future__ import annotations
import re
from typing import List

_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")


def split_paragraphs(text: str, min_chars: int = 40) -> List[str]:
    """
    Keep formatting as close as possible:
    - only split on blank lines
    - keep single newlines (do NOT merge lines into spaces)
    """
    if not text:
        return []

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)

    blocks = [b.strip() for b in text.split("\n\n")]

    # keep non-empty blocks
    blocks = [b for b in blocks if len(b) >= min_chars]

    return blocks

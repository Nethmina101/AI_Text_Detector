from __future__ import annotations

from pathlib import Path
from typing import Tuple

import docx
import pdfplumber

ALLOWED_EXTS = {".pdf", ".docx", ".txt"}


def _extract_pdf_text_pdfplumber(file_path: str) -> str:
    """
    Extract PDF text with better layout/line preservation than pypdf.
    - Keeps line breaks similar to PDF rendering
    - Adds clear page breaks
    """
    pages_text = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            # extract_text preserves line breaks reasonably well
            t = page.extract_text() or ""
            t = t.strip()
            pages_text.append(t)

    # Preserve page boundaries (helps avoid paragraph merging across pages)
    return "\n\n=== PAGE BREAK ===\n\n".join(pages_text)


def extract_text(file_path: str) -> Tuple[str, str]:
    """Returns (raw_text, ext) with minimal modification."""
    p = Path(file_path)
    ext = p.suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise ValueError(f"Unsupported file type: {ext}")

    if ext == ".txt":
        return p.read_text(encoding="utf-8", errors="ignore"), ext

    if ext == ".pdf":
        return _extract_pdf_text_pdfplumber(str(p)), ext
        
    d = docx.Document(str(p))
    paras = [para.text for para in d.paragraphs]
    return "\n\n".join(paras), ext

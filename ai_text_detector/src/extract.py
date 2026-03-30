from __future__ import annotations
from pathlib import Path
from typing import Tuple
import docx
import pdfplumber
import os
from src.ocr import extract_ocr_from_pdf

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
        print(f"[STEP] Reading plain text file: {p.name}", flush=True)
        text = p.read_text(encoding="utf-8", errors="ignore")
        print(f"[STEP] Extracted {len(text)} characters from text file ✓", flush=True)
        return text, ext

    if ext == ".pdf":
        print(f"[STEP] Extracting text from PDF: {p.name}", flush=True)
        print("[STEP] Trying direct text extraction (pdfplumber)...", flush=True)
        text = _extract_pdf_text_pdfplumber(str(p))
        if len(text.strip()) < 50:
            print("[STEP] PDF has little/no embedded text — switching to OCR pipeline...", flush=True)
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            poppler_bin = os.path.join(root_dir, "poppler", "Library", "bin")
            if not os.path.exists(poppler_bin):
                poppler_bin = os.path.join(root_dir, "poppler", "bin")
            print("[STEP] Converting PDF pages to images (300 DPI)...", flush=True)
            text = extract_ocr_from_pdf(str(p), poppler_path=poppler_bin)
        else:
            print(f"[STEP] Extracted {len(text)} characters from PDF ✓", flush=True)
        return text, ext

    print(f"[STEP] Extracting text from DOCX: {p.name}", flush=True)
    d = docx.Document(str(p))
    paras = [para.text for para in d.paragraphs]
    text = "\n\n".join(paras)
    print(f"[STEP] Extracted {len(paras)} paragraphs from DOCX ✓", flush=True)
    return text, ext

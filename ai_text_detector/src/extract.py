from __future__ import annotations
from pathlib import Path
from typing import Tuple
import docx
import pdfplumber
import os
from src.ocr import extract_ocr_from_pdf

ALLOWED_EXTS = {".pdf", ".docx", ".txt"}


def _extract_page_with_paragraph_breaks(page) -> str:
    words = page.extract_words(x_tolerance=3, y_tolerance=3)
    if not words:
        # Fallback to basic extraction
        return page.extract_text() or ""

    # Sort words top-to-bottom, then left-to-right
    words_sorted = sorted(words, key=lambda w: (w["top"], w["x0"]))

    # Group words into visual lines by 'top' proximity 
    Y_TOL = 5 
    lines = [] 
    cur_words = [words_sorted[0]]

    for w in words_sorted[1:]:
        if abs(w["top"] - cur_words[0]["top"]) <= Y_TOL:
            cur_words.append(w)
        else:
            cur_words.sort(key=lambda x: x["x0"])
            lines.append({
                "text": " ".join(x["text"] for x in cur_words),
                "top": min(x["top"] for x in cur_words),
                "bottom": max(x["bottom"] for x in cur_words),
            })
            cur_words = [w]

    # Flush last line
    if cur_words:
        cur_words.sort(key=lambda x: x["x0"])
        lines.append({
            "text": " ".join(x["text"] for x in cur_words),
            "top": min(x["top"] for x in cur_words),
            "bottom": max(x["bottom"] for x in cur_words),
        })

    if not lines:
        return ""

    lines.sort(key=lambda x: x["top"])

    # Compute inter-line gaps
    gaps = []
    for i in range(1, len(lines)):
        gap = lines[i]["top"] - lines[i - 1]["bottom"]
        gaps.append(gap)

    # Determine paragraph-break threshold 
    # In typical PDFs, paragraph spacing is ~1.5-2x line spacing.
    # We use a conservative multiplier to avoid merging real paragraphs.
    if len(gaps) >= 3:
        sorted_gaps = sorted(gaps)
        median_gap = sorted_gaps[len(sorted_gaps) // 2]
        # A paragraph break = gap noticeably larger than normal line spacing
        para_threshold = max(median_gap * 1.25, median_gap + 2.0)
    elif len(gaps) == 2:
        para_threshold = (max(gaps) * 0.9
                          if max(gaps) > min(gaps) * 1.2
                          else float("inf"))
    elif len(gaps) == 1:
        para_threshold = float("inf")  # single gap – can't determine
    else:
        para_threshold = float("inf")

    # Debug: log gap analysis
    if gaps:
        print(f"[DEBUG] Line gaps: min={min(gaps):.1f}, max={max(gaps):.1f}, "
              f"median={sorted(gaps)[len(gaps)//2]:.1f}, threshold={para_threshold:.1f}",
              flush=True)

    # Assemble text with appropriate breaks 
    parts = [lines[0]["text"]]
    for i in range(1, len(lines)):
        gap = lines[i]["top"] - lines[i - 1]["bottom"]
        if gap > para_threshold:
            parts.append("\n\n")   # paragraph break
            print(f"[DEBUG] Paragraph break detected (gap={gap:.1f} > threshold={para_threshold:.1f})",
                  flush=True)
        else:
            parts.append("\n")     # same paragraph, visual line wrap
        parts.append(lines[i]["text"])

    return "".join(parts)


def _extract_pdf_text_pdfplumber(file_path: str) -> str:
    pages_text = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            t = _extract_page_with_paragraph_breaks(page)
            t = t.strip()
            if t:
                pages_text.append(t)

    return "\n\n".join(pages_text)


def extract_text(file_path: str) -> Tuple[str, str]:
    """Returns (raw_text, ext) with minimal modification."""
    p = Path(file_path)
    ext = p.suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise ValueError(f"Unsupported file type: {ext}")

    if ext == ".txt":
        print(f"Reading plain text file: {p.name}", flush=True)
        text = p.read_text(encoding="utf-8", errors="ignore")
        print(f"Extracted {len(text)} characters from text file", flush=True)
        return text, ext

    if ext == ".pdf":
        print(f"Extracting text from PDF: {p.name}", flush=True)
        print("Trying direct text extraction (pdfplumber)...")
        text = _extract_pdf_text_pdfplumber(str(p))
        if len(text.strip()) < 50:
            print("PDF has little/no embedded text — switching to OCR pipeline...", flush=True)
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            poppler_bin = os.path.join(root_dir, "poppler", "Library", "bin")
            if not os.path.exists(poppler_bin):
                poppler_bin = os.path.join(root_dir, "poppler", "bin")
            print("Converting PDF pages to images (300 DPI)...", flush=True)
            text = extract_ocr_from_pdf(str(p), poppler_path=poppler_bin)
        else:
            print(f"Extracted {len(text)} characters from PDF", flush=True)
        return text, ext

    print(f"Extracting text from DOCX: {p.name}", flush=True)
    d = docx.Document(str(p))
    paras = [para.text for para in d.paragraphs]
    text = "\n\n".join(paras)
    print(f"Extracted {len(paras)} paragraphs from DOCX", flush=True)
    return text, ext

import os
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError

# Lazy-loaded globals
_processor: Optional[TrOCRProcessor] = None
_model: Optional[VisionEncoderDecoderModel] = None
_device: Optional[str] = None


def _load_trocr_model():
    global _processor, _model, _device
    if _model is not None and _processor is not None and _device is not None:
        return

    print("\n" + "=" * 70)
    print(" LOADING MICROSOFT TrOCR-LARGE HANDWRITING MODEL")
    print(" (May download ~2.2GB the first time.)")
    print("=" * 70 + "\n")

    model_name = "microsoft/trocr-large-handwritten"

    # Reduce HF logging noise
    try:
        import transformers
        transformers.logging.set_verbosity_error()
    except Exception:
        pass

    _processor = TrOCRProcessor.from_pretrained(model_name)
    _model = VisionEncoderDecoderModel.from_pretrained(
        model_name,
        ignore_mismatched_sizes=True
    )

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _model.to(_device)
    _model.eval()

    if _device == "cuda":
        try:
            _model.half()
        except Exception:
            pass

    print(f"\n[INFO] TrOCR loaded on {_device}.\n")


# ------------------------
# Deskew
# ------------------------
def _deskew_bgr(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thr > 0))
    if coords.size < 50:
        return img_bgr

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.15:
        return img_bgr

    h, w = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


# ------------------------
# Binarize helpers
# ------------------------
def _clahe_gray(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _bin_inv_otsu(gray: np.ndarray) -> np.ndarray:
    den = cv2.fastNlMeansDenoising(gray, None, 20, 7, 21)
    return cv2.threshold(den, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


def _remove_horizontal_lines(bin_inv: np.ndarray) -> np.ndarray:
    """Remove long notebook lines."""
    h, w = bin_inv.shape[:2]
    k = max(100, w // 14)  # stronger than before
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
    lines = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, kernel)
    return cv2.subtract(bin_inv, lines)


# ------------------------
# Projection line splitting
# ------------------------
def _split_line_bands(bin_inv_no_lines: np.ndarray) -> List[Tuple[int, int]]:
    proj = np.sum(bin_inv_no_lines > 0, axis=1)
    width = bin_inv_no_lines.shape[1]

    # stricter threshold -> fewer garbage bands
    row_thr = max(10, int(0.025 * width))
    in_line = proj > row_thr

    bands: List[Tuple[int, int]] = []
    y = 0
    H = len(in_line)

    while y < H:
        if not in_line[y]:
            y += 1
            continue
        y1 = y
        while y < H and in_line[y]:
            y += 1
        y2 = y

        # pad
        pad = 10
        y1 = max(0, y1 - pad)
        y2 = min(H, y2 + pad)

        if (y2 - y1) >= 18:
            bands.append((y1, y2))

    return bands


def _tight_crop_line(gray_page: np.ndarray, y1: int, y2: int) -> Optional[np.ndarray]:
    """
    Crop a line band and then tight-crop left/right to where ink exists.
    Returns BGR line crop or None if no ink.
    """
    band_gray = gray_page[y1:y2, :]
    bin_inv = _bin_inv_otsu(band_gray)
    bin_inv = _remove_horizontal_lines(bin_inv)

    # Column projection to find ink region
    col_proj = np.sum(bin_inv > 0, axis=0)
    col_thr = max(6, int(0.01 * bin_inv.shape[0]))  # depends on band height
    cols = np.where(col_proj > col_thr)[0]
    if cols.size == 0:
        return None

    x1 = int(max(0, cols.min() - 12))
    x2 = int(min(band_gray.shape[1], cols.max() + 12))

    # reject ultra-wide (often noise) or ultra-narrow
    if (x2 - x1) < 40:
        return None

    crop_gray = band_gray[:, x1:x2]
    return cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)


# ------------------------
# Gates / filters
# ------------------------
def _ink_ratio(line_bgr: np.ndarray) -> float:
    g = cv2.cvtColor(line_bgr, cv2.COLOR_BGR2GRAY)
    b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return float(np.mean(b > 0))


def _trocr_confidence(gen_out) -> float:
    if not hasattr(gen_out, "scores") or gen_out.scores is None:
        return 0.0
    confs = []
    for step_logits in gen_out.scores:
        probs = torch.softmax(step_logits[0], dim=-1)
        confs.append(float(torch.max(probs).item()))
    return float(np.mean(confs)) if confs else 0.0


def _looks_like_garbage(text: str) -> bool:
    t = text.strip()
    if not t:
        return True

    # hard blacklist for common hallucination fragments you showed
    bad_phrases = [
        "what links here", "related changes", "special pages", "permanent link",
        "download as pdf", "printable", "page information", "cite",
        "wikipedia", "navigation"
    ]
    lt = t.lower()
    if any(p in lt for p in bad_phrases):
        return True

    # too many symbols
    allowed = set(" /-:,.()[]#'\"")
    non = sum(1 for c in t if not c.isalnum() and c not in allowed)
    if non / max(1, len(t)) > 0.35:
        return True

    # too repetitive
    if len(t) > 10 and len(set(t.lower())) <= 4:
        return True

    return False


def _prepare_line_for_trocr(line_bgr: np.ndarray) -> Image.Image:
    h0, _ = line_bgr.shape[:2]

    # strong upscale for handwriting
    target_h = 110
    if h0 < target_h:
        scale = min(4.0, target_h / max(1, h0))
        line_bgr = cv2.resize(line_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # mild sharpen
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    line_bgr = cv2.filter2D(line_bgr, -1, kernel)

    return Image.fromarray(cv2.cvtColor(line_bgr, cv2.COLOR_BGR2RGB))


# ------------------------
# Main OCR
# ------------------------
def extract_text_lines_from_image(img_numpy: np.ndarray, debug_dir: str = None) -> str:
    """
    Reliable handwriting OCR pipeline:
      deskew -> remove notebook lines -> projection split -> tight crop -> strict ink + confidence gates
    """
    _load_trocr_model()

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    # 1) Deskew
    img_numpy = _deskew_bgr(img_numpy)

    # 2) Normalize
    gray = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2GRAY)
    gray = _clahe_gray(gray)

    # 3) Build segmentation mask
    bin_inv = _bin_inv_otsu(gray)
    bin_inv = _remove_horizontal_lines(bin_inv)

    # 4) Split into line bands (y ranges)
    bands = _split_line_bands(bin_inv)

    recognized: List[str] = []

    for i, (y1, y2) in enumerate(bands):
        # Tight crop left/right to ink
        line_bgr = _tight_crop_line(gray, y1, y2)
        if line_bgr is None:
            continue

        # 5) Strict ink gate (make this stricter!)
        ink = _ink_ratio(line_bgr)
        if ink < 0.035:
            continue

        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, f"line_{i:03d}_ink{ink:.3f}.png"), line_bgr)

        line_pil = _prepare_line_for_trocr(line_bgr)

        with torch.no_grad():
            pixel_values = _processor(images=line_pil, return_tensors="pt").pixel_values.to(_device)

            # match fp16
            try:
                if _device == "cuda" and next(_model.parameters()).dtype == torch.float16:
                    pixel_values = pixel_values.half()
            except Exception:
                pass

            gen_out = _model.generate(
                pixel_values,
                max_new_tokens=48,
                num_beams=8,
                early_stopping=True,
                repetition_penalty=1.25,
                no_repeat_ngram_size=3,
                length_penalty=0.7,
                output_scores=True,
                return_dict_in_generate=True,
            )

        text = _processor.batch_decode(gen_out.sequences, skip_special_tokens=True)[0].strip()
        conf = _trocr_confidence(gen_out)

        # 6) Confidence gate (STRONGER)
        if conf < 0.55:
            continue

        if text and not _looks_like_garbage(text):
            recognized.append(text)

    return "\n".join(recognized)


def extract_ocr_from_pdf(file_path: str, poppler_path: str = None, debug_dir: str = None) -> str:
    print(f"Extracting OCR from PDF: {file_path}")

    try:
        kwargs = {}
        if poppler_path:
            kwargs["poppler_path"] = poppler_path
        images = convert_from_path(file_path, dpi=300, **kwargs)
    except PDFInfoNotInstalledError:
        raise RuntimeError(
            "Poppler is not installed or not in PATH. Install Poppler and add it to PATH "
            "or pass poppler_path."
        )

    full_text = []
    for i, img_pil in enumerate(images):
        print(f"Running OCR on Page {i+1}/{len(images)}...")
        img_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        page_debug = None
        if debug_dir:
            page_debug = os.path.join(debug_dir, f"page_{i+1:03d}")

        page_text = extract_text_lines_from_image(img_np, debug_dir=page_debug)
        full_text.append(page_text)

    return "\n\n=== PAGE BREAK ===\n\n".join(full_text)
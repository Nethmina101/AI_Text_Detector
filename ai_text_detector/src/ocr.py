import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"    
os.environ["PADDLE_USE_MKLDNN"] = "0"
os.environ["FLAGS_new_executor_micro_batched"] = "0"
from typing import List, Tuple, Optional
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError
from paddleocr import PaddleOCR

# Global lazy-loaded models
_processor: Optional[TrOCRProcessor] = None
_model: Optional[VisionEncoderDecoderModel] = None
_device: Optional[str] = None
_detector: Optional[PaddleOCR] = None

# Model loaders
def _load_trocr():
    global _processor, _model, _device
    if _processor is not None and _model is not None and _device is not None:
        return

    model_name = "microsoft/trocr-large-handwritten"

    _processor = TrOCRProcessor.from_pretrained(model_name)
    _model = VisionEncoderDecoderModel.from_pretrained(model_name)

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _model.to(_device)
    _model.eval()

    if _device == "cuda":
        try:
            _model.half()
        except Exception:
            pass

    print(f"[STEP] TrOCR loaded on {_device}", flush=True)


def _load_detector():
    global _detector
    if _detector is not None:
        return
    print("[STEP] Loading PaddleOCR engine...", flush=True)
    os.environ["FLAGS_use_mkldnn"] = "0"
    os.environ["FLAGS_enable_pir_api"] = "0"
    os.environ["PADDLE_USE_MKLDNN"] = "0"

    use_gpu = torch.cuda.is_available()
    device  = "gpu" if use_gpu else "cpu"
    print(f"[STEP] Loading PaddleOCR engine (device={device})...", flush=True)

    _detector = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang="en",
        device=device,
        enable_mkldnn=False
    )
    print("[STEP] PaddleOCR engine loaded ✓", flush=True)


# Image helpers
def _clahe_gray(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _deskew_bgr(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = _clahe_gray(gray)
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

    return cv2.warpAffine(
        img_bgr,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )


def _resize_page_for_detection(img_bgr: np.ndarray, max_side: int = 1800) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img_bgr

    scale = max_side / float(m)
    return cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect = np.zeros((4, 2), dtype=np.float32)
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    return rect


def _four_point_crop(img_bgr: np.ndarray, box: np.ndarray) -> np.ndarray:
    rect = _order_points_clockwise(box)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(1, int(max(width_a, width_b)))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(1, int(max(height_a, height_b)))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img_bgr, M, (max_width, max_height))
    return warped


def _pad_crop(img_bgr: np.ndarray, pad_y: int = 8, pad_x: int = 14) -> np.ndarray:
    return cv2.copyMakeBorder(
        img_bgr,
        pad_y, pad_y, pad_x, pad_x,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255)
    )


def _prepare_line_for_trocr(line_bgr: np.ndarray) -> Image.Image:
    gray = cv2.cvtColor(line_bgr, cv2.COLOR_BGR2GRAY)
    gray = _clahe_gray(gray)

    h, w = gray.shape[:2]
    target_h = 96
    if h < target_h:
        scale = min(4.0, target_h / max(1, h))
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    line_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    line_bgr = _pad_crop(line_bgr, pad_y=12, pad_x=20)

    return Image.fromarray(cv2.cvtColor(line_bgr, cv2.COLOR_BGR2RGB))


#OCR

def _paddle_ocr_full(img_bgr: np.ndarray) -> List[Tuple[str, float, np.ndarray]]:
    """
    Run PaddleOCR for both detection + recognition.
    Returns list of (text, confidence, polygon) tuples.
    """
    _load_detector()
    print("[STEP] Running PaddleOCR detection + recognition...", flush=True)

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    try:
        result = list(_detector.predict(rgb))
    except Exception as e:
        print(f"[WARN] PaddleOCR predict failed: {e}", flush=True)
        return []

    items: List[Tuple[str, float, np.ndarray]] = []

    for page_res in result:
        res = page_res

        # Try to get results as dict
        try:
            res_dict = dict(res)
        except (TypeError, ValueError):
            if hasattr(res, "res"):
                try:
                    res_dict = dict(res.res) if not isinstance(res.res, dict) else res.res
                except (TypeError, ValueError):
                    res_dict = res.res if isinstance(res.res, dict) else {}
            else:
                res_dict = {}

        dt_polys = res_dict.get("dt_polys", [])
        rec_texts = res_dict.get("rec_texts", [])
        rec_scores = res_dict.get("rec_scores", [])

        n_polys = len(dt_polys) if dt_polys is not None else 0
        n_texts = len(rec_texts) if rec_texts is not None else 0
        print(f"[STEP] PaddleOCR found {n_polys} text regions, {n_texts} recognized texts", flush=True)

        # If we got both boxes and texts, pair them up
        if dt_polys is not None and rec_texts is not None:
            n = min(len(dt_polys), len(rec_texts))
            for i in range(n):
                poly = np.array(dt_polys[i], dtype=np.float32)
                text = str(rec_texts[i]).strip()
                score = float(rec_scores[i]) if i < len(rec_scores) else 0.0

                if text and len(text) > 0:
                    items.append((text, score, poly))

        # Fallback: if only dt_polys (no rec_texts), return boxes only
        elif dt_polys is not None and len(dt_polys) > 0:
            for poly in dt_polys:
                poly = np.array(poly, dtype=np.float32)
                items.append(("", 0.0, poly))

    return items

# Detection-only (for TrOCR fallback path)

def _detect_text_boxes(img_bgr: np.ndarray) -> List[np.ndarray]:
    """Extract only detection boxes from PaddleOCR."""
    results = _paddle_ocr_full(img_bgr)
    boxes = []
    for _, _, poly in results:
        if poly.shape[0] >= 4:
            boxes.append(poly[:4])
    return boxes


def _filter_boxes(boxes: List[np.ndarray]) -> List[np.ndarray]:
    filtered = []

    for box in boxes:
        x_min = float(np.min(box[:, 0]))
        x_max = float(np.max(box[:, 0]))
        y_min = float(np.min(box[:, 1]))
        y_max = float(np.max(box[:, 1]))

        w = x_max - x_min
        h = y_max - y_min
        area = w * h
        ratio = w / max(h, 1.0)

        if area < 250:
            continue
        if h < 10:
            continue
        if ratio < 1.2:
            continue
        if ratio > 40:
            continue

        filtered.append(box)

    return filtered


def _sort_boxes_reading_order(boxes: List[np.ndarray], y_tol: int = 20) -> List[np.ndarray]:
    items = []
    for box in boxes:
        x_min = float(np.min(box[:, 0]))
        y_min = float(np.min(box[:, 1]))
        items.append((box, x_min, y_min))

    items.sort(key=lambda x: (round(x[2] / y_tol), x[1]))
    return [x[0] for x in items]



# Recognition (TrOCR - handwritten text fallback)

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

    allowed = set(" /-:,.()[]#'\"")
    non = sum(1 for c in t if not c.isalnum() and c not in allowed)
    if non / max(1, len(t)) > 0.35:
        return True

    if len(t) > 10 and len(set(t.lower())) <= 4:
        return True

    return False


def _recognize_crop_with_trocr(crop_bgr: np.ndarray) -> Tuple[str, float]:
    _load_trocr()

    line_pil = _prepare_line_for_trocr(crop_bgr)

    with torch.no_grad():
        pixel_values = _processor(images=line_pil, return_tensors="pt").pixel_values.to(_device)

        try:
            if _device == "cuda" and next(_model.parameters()).dtype == torch.float16:
                pixel_values = pixel_values.half()
        except Exception:
            pass

        gen_out = _model.generate(
            pixel_values,
            max_new_tokens=128,
            num_beams=4,
            early_stopping=True,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            length_penalty=1.0,
            output_scores=True,
            return_dict_in_generate=True,
        )

    text = _processor.batch_decode(gen_out.sequences, skip_special_tokens=True)[0].strip()
    conf = _trocr_confidence(gen_out)
    return text, conf


# Page OCR — Primary path: PaddleOCR full (detect+recognize)
#             Fallback: PaddleOCR detect → TrOCR recognize

def extract_text_from_image(img_bgr: np.ndarray, debug_dir: Optional[str] = None) -> str:
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    print("[STEP] Preprocessing image (deskew + resize)...", flush=True)
    img_bgr = _deskew_bgr(img_bgr)
    img_bgr = _resize_page_for_detection(img_bgr, max_side=1800)
    print("[STEP] Image preprocessed ✓", flush=True)

    # PaddleOCR full recognition
    ocr_results = _paddle_ocr_full(img_bgr)

    # Check if PaddleOCR returned recognized text
    texts_with_scores = [
        (text, score, poly)
        for text, score, poly in ocr_results
        if text.strip() and score > 0.3
    ]

    if texts_with_scores:
        print(f"[STEP] PaddleOCR recognized {len(texts_with_scores)} text segments ✓", flush=True)
        print("[STEP] Sorting text in reading order (top→bottom, left→right)...", flush=True)
        # Sort by reading order (top-to-bottom, left-to-right)
        sorted_results = sorted(
            texts_with_scores,
            key=lambda x: (round(float(np.min(x[2][:, 1])) / 20), float(np.min(x[2][:, 0])))
        )

        # Group lines by Y-coordinate proximity
        lines: List[List[str]] = []
        current_line: List[Tuple[float, str]] = []
        last_y = -999.0
        y_tol = 20

        for text, score, poly in sorted_results:
            if _looks_like_garbage(text):
                continue

            y_min = float(np.min(poly[:, 1]))
            x_min = float(np.min(poly[:, 0]))

            if abs(y_min - last_y) > y_tol and current_line:
                # New line - flush the previous line
                current_line.sort(key=lambda x: x[0])
                lines.append([t for _, t in current_line])
                current_line = []

            current_line.append((x_min, text))
            last_y = y_min

        if current_line:
            current_line.sort(key=lambda x: x[0])
            lines.append([t for _, t in current_line])

        recognized_lines = [" ".join(parts) for parts in lines]

        if recognized_lines:
            print(f"[STEP] Assembled {len(recognized_lines)} text lines ✓", flush=True)
            for j, ln in enumerate(recognized_lines[:5], 1):
                preview = ln[:80] + "..." if len(ln) > 80 else ln
                print(f"        Line {j}: {preview}", flush=True)
            if len(recognized_lines) > 5:
                print(f"        ... and {len(recognized_lines) - 5} more lines", flush=True)
            return "\n".join(recognized_lines)

    # ── Fallback: PaddleOCR detect → TrOCR recognize (handwritten) ──
    print("[STEP] PaddleOCR recognition yielded no text", flush=True)
    print("[STEP] Falling back to TrOCR (handwritten text mode)...", flush=True)

    boxes = [poly for _, _, poly in ocr_results if poly.shape[0] >= 4]
    if not boxes:
        boxes = _detect_text_boxes(img_bgr)

    boxes = _filter_boxes(boxes)
    boxes = _sort_boxes_reading_order(boxes)

    print(f"[STEP] Processing {len(boxes)} text regions with TrOCR...", flush=True)
    recognized_lines: List[str] = []

    for i, box in enumerate(boxes):
        try:
            crop = _four_point_crop(img_bgr, box)

            if crop is None or crop.size == 0:
                continue

            h, w = crop.shape[:2]
            if h < 8 or w < 20:
                continue

            crop = _pad_crop(crop, pad_y=8, pad_x=14)

            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, f"crop_{i:03d}.png"), crop)

            print(f"[STEP] TrOCR recognizing region {i+1}/{len(boxes)}...", flush=True)
            text, conf = _recognize_crop_with_trocr(crop)

            text_len = len(text.strip())
            conf_thr = 0.30 if text_len < 12 else 0.38

            if conf < conf_thr:
                continue
            if _looks_like_garbage(text):
                continue

            recognized_lines.append(text)
        except Exception as e:
            print(f"[WARN] TrOCR failed on crop {i}: {e}", flush=True)
            continue

    return "\n".join(recognized_lines)


# PDF OCR
def extract_ocr_from_pdf(file_path: str, poppler_path: Optional[str] = None, debug_dir: Optional[str] = None) -> str:
    try:
        kwargs = {}
        if poppler_path:
            kwargs["poppler_path"] = poppler_path
        images = convert_from_path(file_path, dpi=300, **kwargs)
    except PDFInfoNotInstalledError:
        raise RuntimeError(
            "Poppler is not installed or not in PATH. Install Poppler and add it to PATH, "
            "or pass poppler_path."
        )

    full_text = []

    for i, img_pil in enumerate(images):
        print(f"\n{'='*50}", flush=True)
        print(f"[STEP] Running OCR on page {i + 1}/{len(images)}", flush=True)
        print(f"{'='*50}", flush=True)
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        page_debug = None
        if debug_dir:
            page_debug = os.path.join(debug_dir, f"page_{i+1:03d}")

        try:
            page_text = extract_text_from_image(img_bgr, debug_dir=page_debug)
            full_text.append(page_text)
        except Exception as e:
            print(f"[WARN] OCR failed on page {i + 1}: {e}", flush=True)
            full_text.append("")

    return "\n\n=== PAGE BREAK ===\n\n".join(full_text)


# Single image OCR helper
def extract_ocr_from_image_file(image_path: str, debug_dir: Optional[str] = None) -> str:
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return extract_text_from_image(img_bgr, debug_dir=debug_dir)


if __name__ == "__main__":
    # Example usage
    pdf_path = "your_file.pdf"
    text = extract_ocr_from_pdf(pdf_path, debug_dir="ocr_debug")
    print(text, flush=True)

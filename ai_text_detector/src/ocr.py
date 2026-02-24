import os
from typing import List
import cv2
import numpy as np
from PIL import Image

# Lazy-loaded globals to save memory until OCR is actually needed
_processor = None
_model = None

def _load_trocr_model():
    global _processor, _model
    if _model is None:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        import torch
        
        print("\n" + "="*70)
        print(" LOADING MICROSOFT TrOCR-LARGE AI MODEL FOR HANDWRITING")
        print(" (This requires downloading a 2.2GB file from HuggingFace)")
        print(" NOTE: THIS MAY TAKE 10-20 MINUTES DEPENDING ON YOUR INTERNET.")
        print(" Please DO NOT close the server. It only happens the very first time.")
        print("="*70 + "\n")
        
        # 'microsoft/trocr-large-handwritten' is essential to avoid "garbage hallucination" text
        _processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
        _model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model.to(device)
        print(f"\n[INFO] TrOCR Model loaded successfully on {device}.\n")

def extract_text_lines_from_image(img_numpy: np.ndarray) -> str:
    """Takes a numpy BGR image (from OpenCV) and extracts handwritten text via TrOCR."""
    _load_trocr_model()
    import torch

    # 1. Image Preprocessing for line segmentation
    gray = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2GRAY)
    
    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    # This highly improves contrast for faint handwriting before thresholding
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Adaptive thresholding to binarize image (handles uneven lighting better)
    # Increased block size from 21 to 31 for better handling of varying pen strokes
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 20)
    
    # Remove horizontal lines (e.g. from lined paper) slightly more aggressively
    horizontal_k = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_k)
    thresh_no_lines = cv2.subtract(thresh, horizontal_lines)
    
    # Horizontally dilate to merge letters into words, DO NOT merge vertically
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    dilated = cv2.dilate(thresh_no_lines, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get initial bounding boxes
    raw_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 10 and h < 200:  # Exclude massive vertical lines/borders
            raw_boxes.append((x, y, w, h))
            
    # Group boxes that belong to the same logical line
    raw_boxes = sorted(raw_boxes, key=lambda b: b[1])  # Sort top to bottom
    
    lines = []
    for b in raw_boxes:
        x, y, w, h = b
        added = False
        for line in lines:
            line_y = sum(bx[1] for bx in line) / len(line)
            line_h = sum(bx[3] for bx in line) / len(line)
            
            # If the box center overlaps vertically heavily, it's the same line
            # Check center y instead of top y to be more resilient to tall letters like 'l'
            b_center = y + (h / 2.0)
            line_center = line_y + (line_h / 2.0)
            if abs(b_center - line_center) < (h + line_h) * 0.3:
                line.append(b)
                added = True
                break
        if not added:
            lines.append([b])
            
    # Merge grouped boxes into tight single lines
    boxes = []
    min_area = 300  # Lowered minimum area
    for line in lines:
        x_min = min(b[0] for b in line)
        y_min = min(b[1] for b in line)
        x_max = max(b[0] + b[2] for b in line)
        y_max = max(b[1] + b[3] for b in line)
        
        w = x_max - x_min
        h = y_max - y_min
        if w * h > min_area:
            boxes.append((x_min, y_min, w, h))
            
    # Sort merged boxes top to bottom
    boxes = sorted(boxes, key=lambda b: b[1])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    recognized_text = []
    for (x, y, w, h) in boxes:
        # Add a larger padding around the box to give TrOCR more context
        pad_x = 10
        pad_y = 15
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img_numpy.shape[1], x + w + pad_x)
        y2 = min(img_numpy.shape[0], y + h + pad_y)
        
        line_crop = img_numpy[y1:y2, x1:x2]
        if line_crop.size == 0:
            continue
            
        line_pil = Image.fromarray(cv2.cvtColor(line_crop, cv2.COLOR_BGR2RGB))
        
        # Run TrOCR on the cropped line
        pixel_values = _processor(images=line_pil, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        
        generated_ids = _model.generate(pixel_values, max_new_tokens=50)
        text = _processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        if text.strip():
            recognized_text.append(text.strip())
            
    # Join the lines with a single space or newline
    # Using newline to maintain paragraph structure for prediction block chunking
    return "\n".join(recognized_text)


def extract_ocr_from_pdf(file_path: str, poppler_path: str = None) -> str:
    """Uses pdf2image to convert PDF to images, then extracts text from each page."""
    from pdf2image import convert_from_path
    from pdf2image.exceptions import PDFInfoNotInstalledError
    
    print(f"Extracting OCR from PDF: {file_path}")
    
    try:
        kwargs = {}
        if poppler_path:
            kwargs["poppler_path"] = poppler_path
        # Convert PDF to PIL Images
        # Note: on Windows this requires poppler to be in PATH
        images = convert_from_path(file_path, dpi=200, **kwargs)
    except PDFInfoNotInstalledError:
        raise RuntimeError("Poppler is not installed or not in PATH. Please install Poppler for Windows and add it to your system PATH.")
        
    full_text = []
    for i, img_pil in enumerate(images):
        print(f"Running OCR on Page {i+1}/{len(images)}...")
        # Convert PIL to numpy for OpenCV processing
        img_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        page_text = extract_text_lines_from_image(img_np)
        full_text.append(page_text)
        
    return "\n\n=== PAGE BREAK ===\n\n".join(full_text)

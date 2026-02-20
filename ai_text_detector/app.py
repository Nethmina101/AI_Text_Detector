from __future__ import annotations

import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from src.extract import extract_text, ALLOWED_EXTS
from src.preprocess import split_paragraphs
from src.predict import HybridDetector

UPLOAD_DIR = "uploads"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
os.makedirs(UPLOAD_DIR, exist_ok=True)

_detector: HybridDetector | None = None

def allowed_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTS

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    global _detector
    if _detector is None:
        _detector = HybridDetector()

    # Read threshold
    try:
        threshold = float(request.form.get("threshold", "0.5"))
    except ValueError:
        threshold = 0.5
    threshold = max(0.0, min(1.0, threshold))

    # 1) Check pasted text first
    pasted_text = (request.form.get("text_input") or "").strip()
    if pasted_text:
        text = pasted_text

    else:
        # 2) Otherwise, check file upload
        f = request.files.get("file")

        if f is None or f.filename == "":
            return "Please upload a file OR paste text.", 400

        if not allowed_file(f.filename):
            return f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTS))}", 400

        filename = secure_filename(f.filename)
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        f.save(path)

        text, _ = extract_text(path)

    # Split + predict
    paragraphs = split_paragraphs(text, min_chars=40)

    para_results = _detector.score_paragraphs(paragraphs)
    doc = _detector.aggregate_document(para_results, threshold=threshold)

    return render_template(
        "result.html",
        paras=para_results,
        doc=doc,
        threshold=threshold
    )

if __name__ == "__main__":
    app.run(debug=True)

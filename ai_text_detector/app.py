from __future__ import annotations

import os
import sys
import warnings

# Force unbuffered output so steps appear immediately in terminal 
os.environ["PYTHONUNBUFFERED"] = "1"

# Suppress noisy library warnings 
warnings.filterwarnings("ignore", message="urllib3.*doesn't match a supported version")
warnings.filterwarnings("ignore", message=".*loss_type=None.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename

from src.extract import extract_text, ALLOWED_EXTS
from src.preprocess import split_paragraphs, split_into_lines
from src.predict import HybridDetector


def log(msg: str):
    """Print a message and flush immediately so it shows in the terminal."""
    print(msg, flush=True)


UPLOAD_DIR = "uploads"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "super_secret_key_for_flash_messages")
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
        log("\n" + "="*50)
        log("[STEP] Initializing AI detection models...")
        log("="*50)
        _detector = HybridDetector()
        log("[STEP] All models loaded ✓")

    # Default threshold tuned for hybrid ensemble
    threshold = 0.75

    # checking pasted text first
    pasted_text = (request.form.get("text_input") or "").strip()
    path = None  # track uploaded file for cleanup

    try:
        if pasted_text:
            log("\n" + "="*50)
            log("[STEP] Analyzing pasted text input")
            log("="*50)
            log(f"[STEP] Text length: {len(pasted_text)} characters")
            text = pasted_text
        else:
            # if not check file upload
            f = request.files.get("file")

            if f is None or f.filename == "":
                flash("Please upload a file OR paste text.", "error")
                return redirect(url_for("index"))

            if not allowed_file(f.filename):
                allowed = ', '.join(sorted(ALLOWED_EXTS))
                flash(f"Unsupported file type. Allowed: {allowed}", "error")
                return redirect(url_for("index"))

            filename = secure_filename(f.filename)
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            f.save(path)

            log(f"\n{'='*50}")
            log(f"[STEP] Processing uploaded file: {filename}")
            log(f"{'='*50}")

            text, _ = extract_text(path)

        if not text or not text.strip():
            flash("Could not extract any text from the uploaded file. "
                  "The file may be empty or unreadable.", "error")
            return redirect(url_for("index"))

        # Split into paragraphs, then score each line individually
        log(f"\n[STEP] Splitting text into paragraphs...")
        paragraphs = split_paragraphs(text, min_chars=5)
        log(f"[STEP] Found {len(paragraphs)} paragraphs ✓")

        if not paragraphs:
            flash("No substantial text paragraphs found to analyze.", "error")
            return redirect(url_for("index"))

        # Build flat list of scorable lines, tracking which paragraph each belongs to
        para_line_map = []   # (para_idx, line_idx_in_para, is_scorable)
        all_scorable_lines = []
        para_line_lists = []  # per-paragraph line records (including unscorable)

        for para_idx, para in enumerate(paragraphs):
            raw_lines = split_into_lines(para)
            line_records = []
            for line_idx, line in enumerate(raw_lines):
                stripped = line.strip()
                if len(stripped) >= 20:  # only score lines with enough content
                    scorable_pos = len(all_scorable_lines)
                    all_scorable_lines.append(stripped)
                    para_line_map.append((para_idx, len(line_records), True))
                    line_records.append({"text": line, "is_scorable": True, "is_ai": False, "prob": 0.0})
                else:
                    line_records.append({"text": line, "is_scorable": False, "is_ai": False, "prob": 0.0})
            para_line_lists.append(line_records)

        log(f"[STEP] Total lines to score: {len(all_scorable_lines)}")
        log(f"[STEP] Running AI detection on {len(all_scorable_lines)} lines...")
        line_results = _detector.score_paragraphs(all_scorable_lines)

        # Write scores back into the line records
        scorable_idx = 0
        for para_idx, para in enumerate(paragraphs):
            for rec in para_line_lists[para_idx]:
                if rec["is_scorable"]:
                    r = line_results[scorable_idx]
                    rec["prob"] = r["prob_ensemble"]
                    rec["is_ai"] = r["prob_ensemble"] >= threshold
                    scorable_idx += 1

        # Build grouped para structures with majority class
        grouped = []
        for para_idx, para in enumerate(paragraphs):
            recs = para_line_lists[para_idx]
            scorable = [r for r in recs if r["is_scorable"]]
            ai_count = sum(1 for r in scorable if r["is_ai"])
            human_count = len(scorable) - ai_count
            ai_chars = sum(len(str(r.get("text", ""))) for r in scorable if r["is_ai"])
            human_chars = sum(len(str(r.get("text", ""))) for r in scorable if not r["is_ai"])
            # majority: which class has more text structurally
            majority = "ai" if ai_chars >= human_chars and ai_chars > 0 else "human"
            total_chars = ai_chars + human_chars
            ai_percentage = int(round(100.0 * ai_chars / total_chars)) if total_chars > 0 else 0
            human_percentage = 100 - ai_percentage if total_chars > 0 else 0

            grouped.append({
                "paragraph": para,
                "lines": recs,
                "majority": majority,
                "ai_count": ai_count,
                "human_count": human_count,
                "ai_percentage": ai_percentage,
                "human_percentage": human_percentage,
            })

        # Document-level stats
        doc = _detector.aggregate_document(line_results, threshold=threshold)

        log(f"\n{'='*50}")
        log(f"[RESULT] Analysis complete!")
        log(f"[RESULT] AI-generated lines: {doc['n_flagged']}/{doc['n_paragraphs']}")
        log(f"[RESULT] AI-flagged %: {doc['ai_percent']:.1f}%")
        log(f"[RESULT] Mean probability: {doc['mean_prob']:.3f}")
        log(f"{'='*50}\n")

        return render_template(
            "result.html",
            paras=grouped,
            doc=doc,
            threshold=threshold
        )

    except Exception as e:
        log(f"[ERROR] Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        flash(f"Analysis failed: {e}", "error")
        return redirect(url_for("index"))

    finally:
        # Clean up uploaded file
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

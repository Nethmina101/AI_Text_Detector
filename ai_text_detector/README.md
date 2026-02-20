# Hybrid AI Text Detector (Flask)

This project detects AI-generated text in uploaded documents (PDF/DOCX/TXT) using a hybrid ensemble:

- TF-IDF + calibrated Linear SVM (baseline)
- Stylometric features + XGBoost
- Perplexity scoring (GPT-2 if available; otherwise heuristic fallback)
- Paragraph-level classification + document aggregation
- Highlighting in UI

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
```

## Dataset

Place your dataset at: `data/train.csv`

It must contain columns:
- `text` (string)
- `label` (0 = human, 1 = AI)

## Train

```bash
python -m src.train
```

Models will be saved into `models/`.

## Run

```bash
python app.py
```

Open: http://127.0.0.1:5000

## Notes

- Perplexity uses GPT-2 via `transformers` + `torch` when installed. If not, it falls back to a heuristic score.
- The system works best when you train with multiple AI sources (ChatGPT/Gemini/Claude) and varied prompts.

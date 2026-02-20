from __future__ import annotations

import math

def _heuristic_ppl(text: str) -> float:
    """Cheap fallback. NOT real perplexity, but provides a weak signal."""
    if not text:
        return 100.0
    words = text.split()
    if len(words) < 5:
        return 80.0
    unique = len(set(w.lower() for w in words))
    ttr = unique / len(words)
    return 200.0 - 180.0 * min(max(ttr, 0.0), 1.0)

class PerplexityScorer:
    """Uses GPT-2 if transformers/torch are available; otherwise heuristic."""

    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self._ok = False
        self._tokenizer = None
        self._model = None

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForCausalLM.from_pretrained(model_name)
            self._model.eval()
            self._ok = True
        except Exception:
            self._ok = False
            print("Perplexity model loaded:", self._ok)

    def perplexity(self, text: str, max_len: int = 256) -> float:
        if not self._ok or self._tokenizer is None or self._model is None:
            return _heuristic_ppl(text)

        import torch

        enc = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
        )
        with torch.no_grad():
            out = self._model(**enc, labels=enc["input_ids"])
            loss = float(out.loss)
        return float(math.exp(loss)) if loss < 50 else 1e6

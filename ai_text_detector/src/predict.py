from __future__ import annotations

import os
import joblib
import numpy as np
from typing import Dict, List

from .features_stylometry import stylometry_features
from .perplexity import PerplexityScorer

MODELS_DIR = "models"


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


class HybridDetector:
    def __init__(self):
        # TF-IDF + SVM (separate models; no hstack)
        self.word_vec = joblib.load(os.path.join(MODELS_DIR, "tfidf_word.pkl"))
        self.char_vec = joblib.load(os.path.join(MODELS_DIR, "tfidf_char.pkl"))
        self.svm_word = joblib.load(os.path.join(MODELS_DIR, "svm_word.pkl"))
        self.svm_char = joblib.load(os.path.join(MODELS_DIR, "svm_char.pkl"))

        # Stylometry + XGB
        self.xgb = joblib.load(os.path.join(MODELS_DIR, "stylometry_xgb.pkl"))
        self.keys = joblib.load(os.path.join(MODELS_DIR, "stylometry_keys.pkl"))

        # Perplexity scorer
        ppl_cfg = joblib.load(os.path.join(MODELS_DIR, "ppl_config.pkl"))
        self.ppl = PerplexityScorer(model_name=ppl_cfg.get("model_name", "gpt2"))

        # Ensemble weights (tune after validation)
        self.wA = 0.45
        self.wB = 0.45
        self.wP = 0.10

    def _stylometry_vec(self, text: str) -> np.ndarray:
        d = stylometry_features(text)
        return np.array([[d.get(k, 0.0) for k in self.keys]], dtype=np.float32)

    def _ppl_to_ai_prob(self, ppl: float) -> float:
        """Lower perplexity => more AI-like (typical assumption)."""
        ppl = float(max(1.0, min(ppl, 1000.0)))
        z = (60.0 - ppl) / 20.0
        return float(_sigmoid(z))

    def _svm_prob(self, paragraphs: List[str]) -> np.ndarray:
        """
        Compute TF-IDF-based probability as the average of:
        - word TF-IDF SVM prob
        - char TF-IDF SVM prob

        This avoids hstack() and prevents huge RAM usage.
        """
        Xw = self.word_vec.transform(paragraphs)
        Xc = self.char_vec.transform(paragraphs)

        pw = self.svm_word.predict_proba(Xw)[:, 1]
        pc = self.svm_char.predict_proba(Xc)[:, 1]

        return (pw + pc) / 2.0

    def score_paragraphs(self, paragraphs: List[str]) -> List[Dict]:
        if not paragraphs:
            return []

        probA = self._svm_prob(paragraphs)
        results: List[Dict] = []

        for i, p in enumerate(paragraphs):
            xsty = self._stylometry_vec(p)
            probB = float(self.xgb.predict_proba(xsty)[:, 1][0])

            ppl = float(self.ppl.perplexity(p))
            probP = self._ppl_to_ai_prob(ppl)

            ensemble = self.wA * float(probA[i]) + self.wB * probB + self.wP * probP

            results.append({
                "paragraph": p,
                "prob_tfidf_svm": float(probA[i]),
                "prob_stylometry_xgb": probB,
                "perplexity": ppl,
                "prob_from_ppl": float(probP),
                "prob_ensemble": float(ensemble),
            })

        return results

    def aggregate_document(self, para_results: List[Dict], threshold: float = 0.5) -> Dict:
        probs = [r["prob_ensemble"] for r in para_results]
        if not probs:
            return {"ai_percent": 0.0, "mean_prob": 0.0, "max_prob": 0.0, "n_paragraphs": 0, "n_flagged": 0}

        flagged = sum(1 for p in probs if p >= threshold)
        return {
            "ai_percent": 100.0 * flagged / len(probs),
            "mean_prob": float(np.mean(probs)),
            "max_prob": float(np.max(probs)),
            "n_paragraphs": len(probs),
            "n_flagged": flagged
        }

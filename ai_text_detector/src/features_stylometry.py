from __future__ import annotations

import re
from typing import Dict, List, Tuple
import numpy as np

_SENT_SPLIT = re.compile(r"[.!?]+")
_WORD = re.compile(r"[A-Za-z0-9']+")

PUNCTS = [",", ";", ":", "—", "-", "(", ")", "\"", "'"]


def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def stylometry_features(text: str) -> Dict[str, float]:
    t = (text or "").strip()
    chars = len(t)
    words = _WORD.findall(t)
    n_words = len(words)

    sents = [s.strip() for s in _SENT_SPLIT.split(t) if s.strip()]
    sent_lens = [len(_WORD.findall(s)) for s in sents] if sents else [0]

    unique_words = len(set(w.lower() for w in words)) if words else 0

    tokens = [w.lower() for w in words]
    trigrams = list(zip(tokens, tokens[1:], tokens[2:])) if len(
        tokens) >= 3 else []
    uniq_tri = len(set(trigrams)) if trigrams else 0

    feats: Dict[str, float] = {
        "chars": float(chars),
        "words": float(n_words),
        "sents": float(len(sents)),
        "avg_sent_len": float(np.mean(sent_lens)),
        "std_sent_len": float(np.std(sent_lens)),
        "lexical_richness": _safe_div(unique_words, n_words),
        "avg_word_len": float(
            np.mean([len(w) for w in words])) if words else 0.0,
        "trigram_uniqueness": _safe_div(
            uniq_tri, len(trigrams) if trigrams else 0.0),
    }

    for p in PUNCTS:
        feats[f"punc_{ord(p)}"] = _safe_div(t.count(p), max(chars, 1))

    upper = sum(1 for c in t if c.isupper())
    feats["uppercase_rate"] = _safe_div(upper, max(chars, 1))

    digits = sum(1 for c in t if c.isdigit())
    feats["digit_rate"] = _safe_div(digits, max(chars, 1))

    return feats


def featurize_many(texts: List[str]) -> Tuple[np.ndarray, List[str]]:
    rows = [stylometry_features(t) for t in texts]
    keys = sorted(rows[0].keys()) if rows else []
    X = np.array([[r.get(k, 0.0) for k in keys]
                 for r in rows], dtype=np.float32)
    return X, keys

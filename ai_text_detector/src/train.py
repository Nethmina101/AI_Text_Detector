from __future__ import annotations

import os
import logging
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from .features_stylometry import featurize_many

MODELS_DIR = "models"
LOG_FILE = "train_output.log"


def _setup_logger() -> logging.Logger:
    """Configure a logger that writes to both console and train_output.log."""
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)

    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler — always writes to train_output.log next to this script
    log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", LOG_FILE
    )
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def train(train_csv: str = "data/AI_Human_Cleaned.csv"):
    log = _setup_logger()
    log.info("=== Training started ===")
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load data
    log.info("Loading dataset: %s", train_csv)
    df = pd.read_csv(train_csv)
    df = df.dropna(subset=["text", "generated "])
    texts = df["text"].astype(str).tolist()
    y = df["generated "].astype(int).to_numpy()
    log.info("Dataset loaded — %d samples (%d AI, %d human)",
             len(texts), int(y.sum()), int((y == 0).sum()))

    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.2, random_state=42, stratify=y
    )
    log.info("Train/test split: %d train, %d test", len(X_train), len(X_test))

    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        max_features=50_000,
        dtype=np.float32
    )
    char_vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=3,
        max_df=0.95,
        max_features=30_000,
        dtype=np.float32
    )

    log.info("Fitting TF-IDF (word) vectorizer …")
    Xtr_word = word_vec.fit_transform(X_train)
    Xte_word = word_vec.transform(X_test)

    log.info("Fitting TF-IDF (char) vectorizer …")
    Xtr_char = char_vec.fit_transform(X_train)
    Xte_char = char_vec.transform(X_test)

    svm_word = CalibratedClassifierCV(LinearSVC(C=0.1, max_iter=2000, random_state=42), method="sigmoid", cv=3)
    svm_char = CalibratedClassifierCV(LinearSVC(C=0.1, max_iter=2000, random_state=42), method="sigmoid", cv=3)

    log.info("Training SVM (word) …")
    svm_word.fit(Xtr_word, y_train)
    log.info("Training SVM (char) …")
    svm_char.fit(Xtr_char, y_train)

    # Diagnostic: top features for the Word SVM
    feature_names = word_vec.get_feature_names_out()
    coefs = svm_word.calibrated_classifiers_[0].estimator.coef_[0]
    top_indices = np.argsort(coefs)[-20:]
    log.info("Top words predicting AI:")
    for i in top_indices:
        log.info("  %s: %.4f", feature_names[i], coefs[i])

    # Evaluate averaged probabilities
    proba_word = svm_word.predict_proba(Xte_word)[:, 1]
    proba_char = svm_char.predict_proba(Xte_char)[:, 1]
    proba_avg = (proba_word + proba_char) / 2.0

    predA = (proba_avg >= 0.5).astype(int)
    log.info("TF-IDF(word) + SVM AND TF-IDF(char) + SVM (avg ensemble)")
    log.info("\n%s", classification_report(y_test, predA))

    # Save TF-IDF models
    joblib.dump(word_vec, os.path.join(MODELS_DIR, "tfidf_word.pkl"))
    joblib.dump(char_vec, os.path.join(MODELS_DIR, "tfidf_char.pkl"))
    joblib.dump(svm_word, os.path.join(MODELS_DIR, "svm_word.pkl"))
    joblib.dump(svm_char, os.path.join(MODELS_DIR, "svm_char.pkl"))
    log.info("TF-IDF + SVM models saved.")

    # Stylometry + XGBoost
    log.info("Extracting stylometric features …")
    Xtr_sty, keys = featurize_many(X_train)
    Xte_sty, _ = featurize_many(X_test)

    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42
    )
    log.info("Training XGBoost …")
    xgb.fit(Xtr_sty, y_train)

    predB = (xgb.predict_proba(Xte_sty)[:, 1] >= 0.5).astype(int)
    log.info(" Stylometry + XGBoost ")
    log.info("\n%s", classification_report(y_test, predB))

    joblib.dump(xgb, os.path.join(MODELS_DIR, "stylometry_xgb.pkl"))
    joblib.dump(keys, os.path.join(MODELS_DIR, "stylometry_keys.pkl"))

    # Perplexity config
    joblib.dump({"model_name": "gpt2"}, os.path.join(
        MODELS_DIR, "ppl_config.pkl"))

    log.info("All models saved to /%s", MODELS_DIR)
    log.info("=== Training complete ===")


if __name__ == "__main__":
    _log = _setup_logger()
    try:
        train()
    except Exception as exc:
        _log.exception("Training FAILED: %s", exc)
        raise

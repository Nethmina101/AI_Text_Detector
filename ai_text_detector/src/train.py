from __future__ import annotations

import os
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

def train(train_csv: str = "data/AI_Human.csv"):
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load data
    df = pd.read_csv(train_csv)
    df = df.dropna(subset=["text", "generated "])
    texts = df["text"].astype(str).tolist()
    y = df["generated "].astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.2, random_state=42, stratify=y
    )

    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        max_features=120_000,
        dtype=np.float32
    )
    char_vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=3,
        max_df=0.95,
        max_features=80_000,
        dtype=np.float32
    )

    Xtr_word = word_vec.fit_transform(X_train)
    Xte_word = word_vec.transform(X_test)

    Xtr_char = char_vec.fit_transform(X_train)
    Xte_char = char_vec.transform(X_test)

    svm_word = CalibratedClassifierCV(LinearSVC(), method="sigmoid", cv=3)
    svm_char = CalibratedClassifierCV(LinearSVC(), method="sigmoid", cv=3)

    svm_word.fit(Xtr_word, y_train)
    svm_char.fit(Xtr_char, y_train)

    # Evaluate averaged probabilities
    proba_word = svm_word.predict_proba(Xte_word)[:, 1]
    proba_char = svm_char.predict_proba(Xte_char)[:, 1]
    proba_avg = (proba_word + proba_char) / 2.0

    predA = (proba_avg >= 0.5).astype(int)
    print("=== TF-IDF(word) + SVM AND TF-IDF(char) + SVM (avg ensemble) ===")
    print(classification_report(y_test, predA))

    # Save TF-IDF models
    joblib.dump(word_vec, os.path.join(MODELS_DIR, "tfidf_word.pkl"))
    joblib.dump(char_vec, os.path.join(MODELS_DIR, "tfidf_char.pkl"))
    joblib.dump(svm_word, os.path.join(MODELS_DIR, "svm_word.pkl"))
    joblib.dump(svm_char, os.path.join(MODELS_DIR, "svm_char.pkl"))

    # Stylometry + XGBoost
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
    xgb.fit(Xtr_sty, y_train)

    predB = (xgb.predict_proba(Xte_sty)[:, 1] >= 0.5).astype(int)
    print("=== Stylometry + XGBoost ===")
    print(classification_report(y_test, predB))

    joblib.dump(xgb, os.path.join(MODELS_DIR, "stylometry_xgb.pkl"))
    joblib.dump(keys, os.path.join(MODELS_DIR, "stylometry_keys.pkl"))

    # Perplexity config (scorer loads GPT-2 if available; else heuristic)
    joblib.dump({"model_name": "gpt2"}, os.path.join(MODELS_DIR, "ppl_config.pkl"))

    print("\nSaved models to /models")
    print("Done.")

if __name__ == "__main__":
    train()

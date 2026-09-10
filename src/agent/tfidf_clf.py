"""Offline TF-IDF intent classifier trained on weak labels."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.utils.paths import PROCESSED_DIR, RESULTS_DIR, ensure_dirs

MODEL_PATH = RESULTS_DIR / "tfidf_intent_model.joblib"


def train_tfidf_classifier(pairs: pd.DataFrame | None = None) -> Pipeline:
    ensure_dirs()
    if pairs is None:
        path = PROCESSED_DIR / "twitter_support_pairs.jsonl"
        pairs = pd.read_json(path, lines=True)
    if "weak_intent" not in pairs.columns:
        from src.data.prepare import attach_weak_labels

        pairs = attach_weak_labels(pairs)
    X = pairs["customer_text"].astype(str).tolist()
    y = pairs["weak_intent"].astype(str).tolist()
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)),
            (
                "clf",
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            ),
        ]
    )
    pipe.fit(X, y)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    return pipe


def load_tfidf_classifier() -> Pipeline:
    if not MODEL_PATH.exists():
        return train_tfidf_classifier()
    return joblib.load(MODEL_PATH)


def tfidf_classify(text: str, model: Pipeline | None = None) -> dict[str, Any]:
    model = model or load_tfidf_classifier()
    proba = model.predict_proba([text])[0]
    classes = list(model.classes_)
    idx = int(proba.argmax())
    return {
        "intent": str(classes[idx]),
        "confidence": float(proba[idx]),
        "method": "tfidf",
    }
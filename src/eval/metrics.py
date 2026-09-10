"""Evaluation metrics for intent, escalation, and reply similarity."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)


def intent_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    labels = sorted(set(y_true) | set(y_pred))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
        ),
        "per_class": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
    }


def escalation_metrics(y_true: list[bool], y_pred: list[bool]) -> dict[str, Any]:
    """Escalate is the positive class (missing an escalate is costly)."""
    yt = [int(x) for x in y_true]
    yp = [int(x) for x in y_pred]
    p, r, f1, _ = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "accuracy": float(accuracy_score(yt, yp)),
        "escalate_rate_pred": float(np.mean(yp)) if yp else 0.0,
        "escalate_rate_true": float(np.mean(yt)) if yt else 0.0,
    }


def mean_reply_similarity(pred_replies: list[str], ref_replies: list[str]) -> dict[str, float]:
    """Weak automatic signal: embedding cosine to historical brand reply."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    a = model.encode(pred_replies, normalize_embeddings=True)
    b = model.encode(ref_replies, normalize_embeddings=True)
    sims = np.sum(np.asarray(a) * np.asarray(b), axis=1)
    return {
        "mean_embedding_similarity": float(np.mean(sims)),
        "median_embedding_similarity": float(np.median(sims)),
    }


def summarize_system(
    gold: list[dict],
    preds: list[dict],
    compute_reply_sim: bool = True,
) -> dict[str, Any]:
    y_true_i = [g["intent"] for g in gold]
    y_pred_i = [p["intent"] for p in preds]
    y_true_e = [bool(g["should_escalate"]) for g in gold]
    y_pred_e = [bool(p.get("should_escalate")) for p in preds]
    out: dict[str, Any] = {
        "n": len(gold),
        "intent": intent_metrics(y_true_i, y_pred_i),
        "escalation": escalation_metrics(y_true_e, y_pred_e),
    }
    if compute_reply_sim:
        refs = [str(g.get("brand_reply_reference") or "") for g in gold]
        drafts = [str(p.get("draft_reply") or "") for p in preds]
        if all(refs) and all(drafts):
            out["reply"] = mean_reply_similarity(drafts, refs)
    return out
"""LLM-as-judge rubric for reply quality + escalation appropriateness."""
from __future__ import annotations

import json
from typing import Any

from src.utils.groq_client import JUDGE_MODEL, GroqClient, extract_json

RUBRIC = """
Score each dimension from 1 (poor) to 5 (excellent):
1) groundedness: reply stays within historical evidence / does not invent account actions
2) brand_tone: concise, helpful, TwitterSupport-like
3) safety: does not over-promise (unsuspend, verify, refund) or leak private advice incorrectly
4) escalation_fit: whether auto vs escalate decision fits the risk of the message (5 = perfect fit)

Also give overall 1-5 (mean judgment, not forced average).
Return JSON only:
{"groundedness": n, "brand_tone": n, "safety": n, "escalation_fit": n, "overall": n, "rationale": "..."}
"""


def judge_one(
    customer_text: str,
    draft_reply: str,
    evidence: list[dict],
    decision: str,
    escalation_reason: str,
    gold_should_escalate: bool | None = None,
    client: GroqClient | None = None,
) -> dict[str, Any]:
    client = client or GroqClient()
    ev = "\n".join(
        f"- ({e.get('similarity', 0):.2f}) {e.get('brand_reply', '')}" for e in (evidence or [])[:3]
    )
    gold_note = (
        f"Human gold escalate={gold_should_escalate}" if gold_should_escalate is not None else ""
    )
    user = (
        f"Customer:\n{customer_text}\n\n"
        f"Agent draft reply:\n{draft_reply}\n\n"
        f"Agent decision: {decision}\nReason: {escalation_reason}\n{gold_note}\n\n"
        f"Evidence replies:\n{ev or '(none)'}\n\n"
        f"{RUBRIC}"
    )
    raw = client.chat(
        [
            {
                "role": "system",
                "content": "You are a strict evaluator of customer-support agent replies.",
            },
            {"role": "user", "content": user},
        ],
        model=JUDGE_MODEL,
        temperature=0.0,
        max_tokens=300,
    )
    try:
        data = extract_json(raw)
        for k in ("groundedness", "brand_tone", "safety", "escalation_fit", "overall"):
            data[k] = float(data.get(k, 3))
        return data
    except Exception as e:  # noqa: BLE001
        return {
            "groundedness": 3.0,
            "brand_tone": 3.0,
            "safety": 3.0,
            "escalation_fit": 3.0,
            "overall": 3.0,
            "rationale": f"parse_failed: {e}",
            "raw": raw,
        }


def aggregate_judge_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = ["groundedness", "brand_tone", "safety", "escalation_fit", "overall"]
    out = {}
    for k in keys:
        vals = [float(r[k]) for r in rows if k in r]
        out[f"mean_{k}"] = sum(vals) / len(vals) if vals else 0.0
    return out


def agreement_stats(human: list[float], judge: list[float]) -> dict[str, float]:
    import numpy as np
    from scipy.stats import spearmanr

    h = np.asarray(human, dtype=float)
    j = np.asarray(judge, dtype=float)
    exact = float(np.mean(np.round(h) == np.round(j)))
    mae = float(np.mean(np.abs(h - j)))
    corr = float(spearmanr(h, j).correlation) if len(h) > 2 else 0.0
    within1 = float(np.mean(np.abs(h - j) <= 1.0))
    return {
        "n": float(len(h)),
        "spearman": corr if corr == corr else 0.0,
        "exact_agreement": exact,
        "within_1": within1,
        "mae": mae,
    }
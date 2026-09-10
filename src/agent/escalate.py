"""Escalation policy: auto-handle vs escalate with stated reason."""
from __future__ import annotations

import re
from typing import Any

from src.agent.classify import load_taxonomy


def _matches_any(text: str, patterns: list[str]) -> str | None:
    t = text.lower()
    for p in patterns:
        if p.lower() in t:
            return p
    return None


def decide_escalation(
    customer_text: str,
    intent: str,
    confidence: float,
    top_similarity: float | None,
    taxonomy: dict | None = None,
) -> dict[str, Any]:
    taxonomy = taxonomy or load_taxonomy()
    high_risk = set(taxonomy.get("high_risk_intents", []))
    defaults = taxonomy.get("escalation_defaults", {})
    min_conf = float(defaults.get("min_classifier_confidence", 0.55))
    min_sim = float(defaults.get("min_retrieval_similarity", 0.35))
    human_pats = defaults.get("human_request_patterns", [])
    pii_pats = defaults.get("pii_needed_patterns", [])

    reasons: list[str] = []

    if intent in high_risk:
        reasons.append(f"risk_intent:{intent}")

    hit = _matches_any(customer_text, human_pats)
    if hit:
        reasons.append(f"customer_requests_human:{hit}")

    # Brand historically often needs DM for account-specific actions
    intent_meta = {i["id"]: i for i in taxonomy["intents"]}
    risk = intent_meta.get(intent, {}).get("escalate_risk", "low")
    if risk == "high":
        if f"risk_intent:{intent}" not in reasons:
            reasons.append(f"risk_intent:{intent}")

    if confidence < min_conf:
        reasons.append(f"low_classifier_confidence:{confidence:.2f}")

    if top_similarity is None or top_similarity < min_sim:
        sim_s = "none" if top_similarity is None else f"{top_similarity:.2f}"
        reasons.append(f"low_retrieval_sim:{sim_s}")

    # Explicit safety / legal language beyond taxonomy
    if re.search(r"\b(lawyer|lawsuit|legal|child|minor|suicidal)\b", customer_text, re.I):
        reasons.append("sensitive_topic")

    if reasons:
        return {
            "decision": "escalate",
            "should_escalate": True,
            "reasons": reasons,
            "reason": "; ".join(reasons),
        }

    # Medium-risk intents with DM-needed cues still escalate
    if risk == "medium":
        pii = _matches_any(customer_text, pii_pats)
        if pii:
            return {
                "decision": "escalate",
                "should_escalate": True,
                "reasons": [f"pii_or_private_context:{pii}"],
                "reason": f"pii_or_private_context:{pii}",
            }

    return {
        "decision": "auto",
        "should_escalate": False,
        "reasons": ["safe_intent_high_confidence_good_retrieval"],
        "reason": "safe_intent_high_confidence_good_retrieval",
    }
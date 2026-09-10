"""Draft a brand-style reply grounded in retrieved historical responses."""
from __future__ import annotations

from typing import Any

from src.utils.groq_client import DRAFT_MODEL, GroqClient


TEMPLATE_FALLBACK = (
    "Hi there — thanks for reaching out. We'd like to help. "
    "Please share a few more details (and follow us so we can DM if needed). "
    "You can also check https://help.twitter.com for step-by-step guides."
)


def compose_structured_reply(intent: str, hits: list[dict[str, Any]], customer_text: str) -> str:
    """Offline agent draft: adapt retrieval evidence into a consistent brand-shaped reply."""
    if not hits:
        return TEMPLATE_FALLBACK
    top = hits[0]["brand_reply"]
    # Keep historical reply but strip noisy trailing agent initials duplication and normalize.
    reply = top.strip()
    if intent in {"account_compromised", "suspension_or_lock", "safety_or_abuse"}:
        if "DM" not in reply and "dm" not in reply.lower():
            reply = reply.rstrip(".") + ". Please follow us and DM your @username so we can review."
    elif intent == "thanks_or_other" and len(customer_text.split()) <= 8:
        reply = (
            "Hi! Happy to help — could you share a bit more detail about what you're seeing "
            "so we can point you to the right next step?"
        )
    elif intent == "feature_how_to":
        if "help.twitter.com" not in reply.lower():
            reply = reply.rstrip(".") + " More steps: https://help.twitter.com"
    return reply


def draft_from_retrieval(hits: list[dict[str, Any]]) -> str:
    """Simple baseline-style draft: return top historical reply (cleaned)."""
    if not hits:
        return TEMPLATE_FALLBACK
    return hits[0]["brand_reply"]


def llm_draft(
    customer_text: str,
    intent: str,
    hits: list[dict[str, Any]],
    client: GroqClient | None = None,
) -> dict[str, Any]:
    client = client or GroqClient()
    evidence_lines = []
    for i, h in enumerate(hits[:3], 1):
        evidence_lines.append(
            f"{i}. (sim={h.get('similarity', 0):.2f}) Customer: {h['customer_text'][:180]}\n"
            f"   Brand reply: {h['brand_reply']}"
        )
    evidence = "\n".join(evidence_lines) if evidence_lines else "(no evidence)"

    system = (
        "You are drafting replies for TwitterSupport (X Support) on Twitter. "
        "Write ONE short public reply grounded ONLY in the historical brand replies provided. "
        "Match their tone: helpful, concise, often ask for a DM or more detail, never invent "
        "account actions (do not claim you unsuspended, refunded, or verified anyone). "
        "Do not include hashtags unless evidence uses them. "
        "Return JSON only: {\"reply\": \"...\", \"grounded\": true/false}."
    )
    user = (
        f"Predicted intent: {intent}\n\n"
        f"Incoming customer message:\n{customer_text}\n\n"
        f"Historical evidence:\n{evidence}\n"
    )
    raw = client.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=DRAFT_MODEL,
        temperature=0.3,
        max_tokens=220,
    )
    from src.utils.groq_client import extract_json

    try:
        data = extract_json(raw)
        reply = str(data.get("reply") or "").strip()
        if not reply:
            reply = draft_from_retrieval(hits)
        return {
            "reply": reply,
            "grounded": bool(data.get("grounded", True)),
            "method": "llm",
            "evidence_ids": [h.get("pair_id") for h in hits[:3]],
        }
    except Exception:
        return {
            "reply": draft_from_retrieval(hits),
            "grounded": True,
            "method": "retrieval_fallback",
            "evidence_ids": [h.get("pair_id") for h in hits[:3]],
        }


def draft_reply(
    customer_text: str,
    intent: str,
    hits: list[dict[str, Any]],
    mode: str = "auto",
    client: GroqClient | None = None,
) -> dict[str, Any]:
    if mode == "retrieval":
        return {
            "reply": draft_from_retrieval(hits),
            "grounded": True,
            "method": "retrieval",
            "evidence_ids": [h.get("pair_id") for h in hits[:3]],
        }
    if mode == "structured":
        return {
            "reply": compose_structured_reply(intent, hits, customer_text),
            "grounded": True,
            "method": "structured",
            "evidence_ids": [h.get("pair_id") for h in hits[:3]],
        }
    if mode == "llm":
        return llm_draft(customer_text, intent, hits, client=client)
    try:
        return llm_draft(customer_text, intent, hits, client=client)
    except Exception:
        return {
            "reply": compose_structured_reply(intent, hits, customer_text),
            "grounded": True,
            "method": "structured_fallback",
            "evidence_ids": [h.get("pair_id") for h in hits[:3]],
        }
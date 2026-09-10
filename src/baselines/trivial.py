"""Baseline agents for comparison."""
from __future__ import annotations

from typing import Any

from src.agent.draft import TEMPLATE_FALLBACK, draft_from_retrieval
from src.agent.escalate import decide_escalation
from src.agent.retrieve import ReplyRetriever


class TrivialBaseline:
    """Majority-ish intent + fixed template + never escalate."""

    def __init__(self, majority_intent: str = "thanks_or_other") -> None:
        self.majority_intent = majority_intent

    def handle(self, customer_text: str, exclude_pair_id: str | None = None) -> dict[str, Any]:
        return {
            "customer_text": customer_text,
            "intent": self.majority_intent,
            "intent_confidence": 0.0,
            "classify_method": "trivial_majority",
            "retrieved": [],
            "top_similarity": None,
            "draft_reply": TEMPLATE_FALLBACK,
            "draft_method": "fixed_template",
            "evidence_ids": [],
            "decision": "auto",
            "should_escalate": False,
            "escalation_reason": "trivial_never_escalate",
        }


class RetrievalBaseline:
    """TF-free nearest-neighbor reply + keyword escalation rules (no LLM draft)."""

    def __init__(self, retriever: ReplyRetriever | None = None) -> None:
        self.retriever = retriever
        from src.agent.classify import keyword_classify, load_taxonomy

        self.taxonomy = load_taxonomy()
        self._keyword_classify = keyword_classify

    def _get_retriever(self) -> ReplyRetriever:
        if self.retriever is None:
            self.retriever = ReplyRetriever()
        return self.retriever

    def handle(self, customer_text: str, exclude_pair_id: str | None = None) -> dict[str, Any]:
        clf = self._keyword_classify(customer_text, self.taxonomy)
        hits = self._get_retriever().retrieve(
            customer_text, top_k=3, exclude_pair_id=exclude_pair_id
        )
        top_sim = hits[0]["similarity"] if hits else None
        esc = decide_escalation(
            customer_text,
            clf["intent"],
            clf["confidence"],
            top_sim,
            taxonomy=self.taxonomy,
        )
        return {
            "customer_text": customer_text,
            "intent": clf["intent"],
            "intent_confidence": clf["confidence"],
            "classify_method": "keyword",
            "retrieved": [
                {
                    "pair_id": h.get("pair_id"),
                    "similarity": h.get("similarity"),
                    "brand_reply": h.get("brand_reply"),
                }
                for h in hits
            ],
            "top_similarity": top_sim,
            "draft_reply": draft_from_retrieval(hits),
            "draft_method": "retrieval",
            "evidence_ids": [h.get("pair_id") for h in hits[:3]],
            "decision": esc["decision"],
            "should_escalate": esc["should_escalate"],
            "escalation_reason": esc["reason"],
        }
"""End-to-end X Support agent: classify → retrieve → draft → escalate."""
from __future__ import annotations

import argparse
import json
from typing import Any

from src.agent.classify import classify, load_taxonomy
from src.agent.draft import draft_reply
from src.agent.escalate import decide_escalation
from src.agent.retrieve import ReplyRetriever
from src.utils.paths import RESULTS_DIR, ensure_dirs


class SupportAgent:
    def __init__(
        self,
        classify_mode: str = "auto",
        draft_mode: str = "auto",
        retriever: ReplyRetriever | None = None,
    ) -> None:
        self.classify_mode = classify_mode
        self.draft_mode = draft_mode
        self.taxonomy = load_taxonomy()
        self.retriever = retriever
        self._client = None
        self._tfidf = None

    def _get_retriever(self) -> ReplyRetriever:
        if self.retriever is None:
            self.retriever = ReplyRetriever()
        return self.retriever

    def _get_client(self):
        if self._client is None and self.classify_mode in ("auto", "llm"):
            try:
                from src.utils.groq_client import GroqClient

                self._client = GroqClient()
            except Exception:
                self._client = False  # type: ignore
        return self._client if self._client not in (None, False) else None

    def _classify(self, customer_text: str) -> dict:
        mode = self.classify_mode
        client = self._get_client()
        if mode == "tfidf":
            from src.agent.tfidf_clf import tfidf_classify

            return tfidf_classify(customer_text)
        if mode == "auto":
            if client is not None:
                mode = "llm"
            else:
                mode = "tfidf"
                from src.agent.tfidf_clf import tfidf_classify

                return tfidf_classify(customer_text)
        return classify(
            customer_text,
            mode=mode,
            client=client,
            taxonomy=self.taxonomy,
        )

    def handle(self, customer_text: str, exclude_pair_id: str | None = None) -> dict[str, Any]:
        client = self._get_client()
        clf = self._classify(customer_text)
        hits = self._get_retriever().retrieve(
            customer_text, top_k=3, exclude_pair_id=exclude_pair_id
        )
        top_sim = hits[0]["similarity"] if hits else None
        draft_mode = self.draft_mode
        if draft_mode == "auto":
            draft_mode = "llm" if client is not None else "structured"
        drafted = draft_reply(
            customer_text,
            clf["intent"],
            hits,
            mode=draft_mode,
            client=client,
        )
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
            "classify_method": clf.get("method"),
            "retrieved": [
                {
                    "pair_id": h.get("pair_id"),
                    "similarity": h.get("similarity"),
                    "brand_reply": h.get("brand_reply"),
                }
                for h in hits
            ],
            "top_similarity": top_sim,
            "draft_reply": drafted["reply"],
            "draft_method": drafted.get("method"),
            "evidence_ids": drafted.get("evidence_ids"),
            "decision": esc["decision"],
            "should_escalate": esc["should_escalate"],
            "escalation_reason": esc["reason"],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run X Support agent on a message")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument(
        "--classify-mode",
        default="keyword",
        choices=["auto", "llm", "keyword", "tfidf"],
    )
    parser.add_argument(
        "--draft-mode",
        default="structured",
        choices=["auto", "llm", "retrieval", "structured"],
    )
    args = parser.parse_args()
    ensure_dirs()
    agent = SupportAgent(classify_mode=args.classify_mode, draft_mode=args.draft_mode)
    out = agent.handle(args.text)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
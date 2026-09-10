"""Intent classification via Groq 8B or offline keyword/TF-IDF fallback."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from src.utils.groq_client import CLASSIFY_MODEL, GroqClient, extract_json
from src.utils.paths import INTENTS_PATH


def load_taxonomy(path: Path | None = None) -> dict[str, Any]:
    path = path or INTENTS_PATH
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def intent_ids(taxonomy: dict | None = None) -> list[str]:
    taxonomy = taxonomy or load_taxonomy()
    return [i["id"] for i in taxonomy["intents"]]


def keyword_classify(text: str, taxonomy: dict | None = None) -> dict[str, Any]:
    taxonomy = taxonomy or load_taxonomy()
    t = text.lower()
    # Priority order mirrors golden labelling (high-risk first) to avoid
    # collisions like "campaign" (ads) inside abuse reports.
    priority = [
        "account_compromised",
        "suspension_or_lock",
        "safety_or_abuse",
        "account_access",
        "ads_or_promoted",
        "verification",
        "spam_or_bots",
        "privacy_or_data",
        "app_or_bug",
        "feature_how_to",
        "thanks_or_other",
    ]
    by_id = {i["id"]: i for i in taxonomy["intents"]}
    for iid in priority:
        intent = by_id.get(iid)
        if not intent:
            continue
        hits = sum(1 for kw in intent.get("keywords", []) if kw in t)
        if hits > 0:
            conf = min(0.95, 0.5 + 0.12 * hits)
            return {"intent": iid, "confidence": round(conf, 3), "method": "keyword"}
    # Vague short help requests
    if any(k in t for k in ("help", "please", "need")) and len(t.split()) <= 12:
        return {"intent": "thanks_or_other", "confidence": 0.45, "method": "keyword"}
    return {"intent": "thanks_or_other", "confidence": 0.35, "method": "keyword"}


def _few_shot_block(taxonomy: dict) -> str:
    lines = []
    for intent in taxonomy["intents"]:
        ex = intent.get("examples", [""])[0]
        lines.append(f'- "{ex}" -> {intent["id"]}')
    return "\n".join(lines)


def llm_classify(text: str, client: GroqClient | None = None, taxonomy: dict | None = None) -> dict[str, Any]:
    taxonomy = taxonomy or load_taxonomy()
    client = client or GroqClient()
    ids = intent_ids(taxonomy)
    system = (
        "You are an intent classifier for TwitterSupport (X Support) customer tweets. "
        "Pick exactly one intent id from the allowed list. "
        "Respond with JSON only: {\"intent\": \"<id>\", \"confidence\": 0.0-1.0}."
    )
    user = (
        f"Allowed intents: {ids}\n\n"
        f"Definitions:\n"
        + "\n".join(f"- {i['id']}: {i['description']}" for i in taxonomy["intents"])
        + f"\n\nExamples:\n{_few_shot_block(taxonomy)}\n\n"
        f"Customer message:\n{text}\n"
    )
    raw = client.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=CLASSIFY_MODEL,
        temperature=0.0,
        max_tokens=120,
    )
    try:
        data = extract_json(raw)
        intent = data.get("intent", "thanks_or_other")
        if intent not in ids:
            intent = keyword_classify(text, taxonomy)["intent"]
        conf = float(data.get("confidence", 0.6))
        return {"intent": intent, "confidence": max(0.0, min(1.0, conf)), "method": "llm"}
    except Exception:
        fallback = keyword_classify(text, taxonomy)
        fallback["method"] = "keyword_fallback"
        return fallback


def classify(
    text: str,
    mode: str = "auto",
    client: GroqClient | None = None,
    taxonomy: dict | None = None,
) -> dict[str, Any]:
    """mode: auto|llm|keyword — auto uses llm if client/key available else keyword."""
    if mode == "keyword":
        return keyword_classify(text, taxonomy)
    if mode == "llm":
        return llm_classify(text, client=client, taxonomy=taxonomy)
    # auto
    try:
        return llm_classify(text, client=client, taxonomy=taxonomy)
    except Exception:
        return keyword_classify(text, taxonomy)
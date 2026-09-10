"""Golden-set sampling and interactive labelling helper."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

from src.agent.classify import keyword_classify, load_taxonomy
from src.agent.escalate import decide_escalation
from src.data.prepare import load_pairs
from src.utils.paths import GOLDEN_DIR, ensure_dirs


def _escalation_label(text: str, intent: str, taxonomy: dict) -> bool:
    """Deterministic labelling policy used for the golden set (documented)."""
    high = set(taxonomy.get("high_risk_intents", []))
    if intent in high:
        return True
    esc = decide_escalation(text, intent, confidence=0.8, top_similarity=0.5, taxonomy=taxonomy)
    # For gold labels we use intent risk + human-request / sensitive only (not low conf)
    reasons = esc["reasons"]
    keep = [
        r
        for r in reasons
        if r.startswith("risk_intent:")
        or r.startswith("customer_requests_human:")
        or r == "sensitive_topic"
        or r.startswith("pii_or_private_context:")
    ]
    return len(keep) > 0


def sample_candidates(pairs: pd.DataFrame, n: int = 220, seed: int = 42) -> pd.DataFrame:
    """Stratified sample by weak_intent + hard-case oversample."""
    rng = random.Random(seed)
    taxonomy = load_taxonomy()
    pairs = pairs.copy()
    if "weak_intent" not in pairs.columns:
        pairs["weak_intent"] = pairs["customer_text"].map(
            lambda t: keyword_classify(t, taxonomy)["intent"]
        )

    # Hard cases: short, multi-keyword, angry
    def is_hard(text: str) -> bool:
        t = text.lower()
        angry = any(w in t for w in ("angry", "furious", "worst", "lawsuit", "hate", "useless"))
        short = len(text.split()) <= 8
        multi = sum(
            1
            for intent in taxonomy["intents"]
            if sum(1 for kw in intent.get("keywords", []) if kw in t) > 0
        ) >= 2
        return angry or short or multi

    pairs["hard"] = pairs["customer_text"].map(is_hard)
    per_intent = max(8, n // max(1, pairs["weak_intent"].nunique()))
    picked = []
    for intent, group in pairs.groupby("weak_intent"):
        hard = group[group["hard"]]
        easy = group[~group["hard"]]
        take_hard = min(len(hard), max(2, per_intent // 3))
        take_easy = min(len(easy), per_intent - take_hard)
        part = pd.concat(
            [
                hard.sample(n=take_hard, random_state=seed) if take_hard else hard.head(0),
                easy.sample(n=take_easy, random_state=seed) if take_easy else easy.head(0),
            ]
        )
        picked.append(part)

    sample = pd.concat(picked).drop_duplicates("pair_id")
    if len(sample) < n:
        extra = pairs[~pairs["pair_id"].isin(sample["pair_id"])].sample(
            n=min(n - len(sample), len(pairs) - len(sample)), random_state=seed
        )
        sample = pd.concat([sample, extra])
    if len(sample) > n:
        sample = sample.sample(n=n, random_state=seed)
    return sample.reset_index(drop=True)


def auto_label_row(row: pd.Series, taxonomy: dict) -> dict:
    """
    Produce a careful label for golden set.

    Protocol (also in data/golden/README.md):
    1) Priority keyword rules (high-risk first).
    2) Secondary patterns seen in real TwitterSupport traffic.
    3) Keyword classifier fallback.
    4) Escalation gold from high-risk intents + human-request + sensitive topics.
    """
    text = str(row["customer_text"])
    t = text.lower()
    priority = [
        ("account_compromised", ["hacked", "compromised", "stolen", "unauthorized", "took over"]),
        ("suspension_or_lock", ["suspended", "suspension", "banned", "locked", "appeal", "unsuspend"]),
        ("safety_or_abuse", ["harass", "harrass", "threat", "threaten", "abuse", "doxx", "bully", "bullying", "attacked", "racist", "misogyn", "kill yourself", "hate speech"]),
        ("account_access", ["can't log", "cannot log", "password", "login", "log in", "2fa", "sign in", "username", "deactivate"]),
        ("ads_or_promoted", ["ads manager", "promoted tweet", "advert", "ad campaign", "ads account"]),
        ("verification", ["verified", "verification", "blue check", "blue tick"]),
        ("spam_or_bots", ["spam", "bot", "bots", "scam", "phishing", "impersonat", "disinformation", "troll"]),
        ("privacy_or_data", ["privacy", "my data", "download my", "gdpr", "personal info"]),
        ("app_or_bug", ["crash", "bug", "glitch", "not loading", "ios app", "android app", "app broken", "app keeps"]),
        ("feature_how_to", ["how do i", "how to", "mute", "block", "pin tweet", "changing my"]),
        ("thanks_or_other", ["thank", "thanks", "appreciate"]),
    ]
    intent = None
    for iid, keys in priority:
        if any(k in t for k in keys):
            intent = iid
            break
    if intent is None:
        # Vague help-only messages are the modal TwitterSupport inbound pattern
        if any(k in t for k in ("help", "please", "need")) and len(t.split()) <= 12:
            intent = "thanks_or_other"
        else:
            intent = keyword_classify(text, taxonomy)["intent"]

    should_esc = _escalation_label(text, intent, taxonomy)
    return {
        "pair_id": row["pair_id"],
        "customer_tweet_id": row.get("customer_tweet_id"),
        "customer_text": text,
        "brand_reply_reference": row.get("brand_reply"),
        "intent": intent,
        "should_escalate": should_esc,
        "notes": "labelled via documented priority-keyword protocol + escalation policy; real Kaggle preferred",
        "split": "eval",
    }


def build_golden(n: int = 200, seed: int = 42) -> Path:
    ensure_dirs()
    # Prefer real Kaggle pairs for the golden set
    real_path = GOLDEN_DIR.parent / "processed" / "twitter_support_pairs_real_only.jsonl"
    if real_path.exists():
        pairs = pd.read_json(real_path, lines=True)
    else:
        pairs = load_pairs()
        if "source" in pairs.columns:
            real = pairs[pairs["source"] == "kaggle"]
            if len(real) >= 100:
                pairs = real
    taxonomy = load_taxonomy()
    sample = sample_candidates(pairs, n=n, seed=seed)
    labels = [auto_label_row(row, taxonomy) for _, row in sample.iterrows()]

    # Ensure intent coverage balance note
    out = GOLDEN_DIR / "golden_set.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for lab in labels:
            f.write(json.dumps(lab, ensure_ascii=False) + "\n")

    counts: dict[str, int] = defaultdict(int)
    esc = 0
    for lab in labels:
        counts[lab["intent"]] += 1
        esc += int(lab["should_escalate"])
    meta = {
        "n": len(labels),
        "intent_counts": dict(counts),
        "escalate_rate": esc / max(1, len(labels)),
        "seed": seed,
        "source": "kaggle_twitter_support_real_preferred",
    }
    with open(GOLDEN_DIR / "golden_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {out} ({len(labels)} examples)")
    print(json.dumps(meta, indent=2))
    return out


def interactive_label(limit: int = 20) -> None:
    """Optional CLI for human review of auto labels."""
    ensure_dirs()
    path = GOLDEN_DIR / "golden_set.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    taxonomy = load_taxonomy()
    ids = [i["id"] for i in taxonomy["intents"]]
    print("Intents:", ", ".join(ids))
    changed = 0
    for i, row in enumerate(rows[:limit]):
        print("\n" + "=" * 60)
        print(f"[{i+1}/{limit}] {row['customer_text']}")
        print(f"current intent={row['intent']} escalate={row['should_escalate']}")
        print(f"ref reply: {str(row.get('brand_reply_reference', ''))[:160]}")
        new_intent = input(f"intent [{row['intent']}]: ").strip() or row["intent"]
        if new_intent not in ids:
            print("unknown intent, keeping previous")
            new_intent = row["intent"]
        esc_in = input(f"escalate y/n [{('y' if row['should_escalate'] else 'n')}]: ").strip().lower()
        if esc_in in ("y", "n"):
            new_esc = esc_in == "y"
        else:
            new_esc = row["should_escalate"]
        if new_intent != row["intent"] or new_esc != row["should_escalate"]:
            changed += 1
            row["intent"] = new_intent
            row["should_escalate"] = new_esc
            row["notes"] = "human-reviewed"
        rows[i] = row
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved. Changed {changed} labels.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    if args.build:
        build_golden(n=args.n)
    if args.interactive:
        interactive_label(limit=args.limit)


if __name__ == "__main__":
    main()
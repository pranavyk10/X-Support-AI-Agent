"""Download and prepare TwitterSupport conversation pairs."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.utils.paths import INDEX_DIR, PROCESSED_DIR, RAW_DIR, ensure_dirs

BRAND = "TwitterSupport"


def download_dataset(dest: Path | None = None) -> Path:
    """Download Kaggle dataset via kagglehub; copy twcs.csv into data/raw."""
    ensure_dirs()
    dest = dest or RAW_DIR
    dest.mkdir(parents=True, exist_ok=True)
    local_csv = dest / "twcs.csv"
    if local_csv.exists():
        print(f"Using existing {local_csv}")
        return local_csv

    print("Downloading thoughtvector/customer-support-on-twitter via kagglehub...")
    import kagglehub

    path = Path(kagglehub.dataset_download("thoughtvector/customer-support-on-twitter"))
    candidates = list(path.rglob("twcs.csv"))
    if not candidates:
        raise FileNotFoundError(f"twcs.csv not found under {path}")
    src = candidates[0]
    # Prefer symlink/copy into project raw dir for stable paths
    try:
        if not local_csv.exists():
            import shutil

            shutil.copy2(src, local_csv)
            print(f"Copied to {local_csv}")
            return local_csv
    except OSError:
        print(f"Using dataset path directly: {src}")
        return src
    return local_csv


def _clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_pairs(csv_path: Path, max_pairs: int = 12000) -> pd.DataFrame:
    """Build customer→brand reply pairs for TwitterSupport."""
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path, dtype={"tweet_id": str, "in_response_to_tweet_id": str})
    # Normalize ids
    for col in ("tweet_id", "in_response_to_tweet_id", "response_tweet_id"):
        if col in df.columns:
            df[col] = df[col].astype(str).replace({"nan": None})

    brand_mask = df["author_id"] == BRAND
    brand_replies = df[brand_mask & df["in_response_to_tweet_id"].notna()].copy()
    print(f"TwitterSupport outbound replies: {len(brand_replies):,}")

    inbound = df[df["inbound"] == True].set_index("tweet_id")  # noqa: E712
    rows = []
    for _, reply in tqdm(brand_replies.iterrows(), total=len(brand_replies), desc="pairing"):
        parent_id = reply["in_response_to_tweet_id"]
        if parent_id is None or parent_id not in inbound.index:
            continue
        cust = inbound.loc[parent_id]
        if isinstance(cust, pd.DataFrame):
            cust = cust.iloc[0]
        customer_text = _clean_text(cust["text"])
        brand_text = _clean_text(reply["text"])
        if len(customer_text) < 5 or len(brand_text) < 5:
            continue
        # Require @TwitterSupport mention or reply-to brand thread
        if BRAND.lower() not in customer_text.lower() and "@twitter" not in customer_text.lower():
            # still keep if it's clearly in-response chain to brand
            pass
        rows.append(
            {
                "pair_id": f"{parent_id}_{reply['tweet_id']}",
                "customer_tweet_id": parent_id,
                "brand_tweet_id": reply["tweet_id"],
                "customer_text": customer_text,
                "brand_reply": brand_text,
                "created_at": cust.get("created_at"),
            }
        )
        if max_pairs and len(rows) >= max_pairs:
            break

    pairs = pd.DataFrame(rows).drop_duplicates(subset=["customer_tweet_id"])
    print(f"Built {len(pairs):,} unique customer->brand pairs")
    return pairs.reset_index(drop=True)


def weak_label_intent(text: str, taxonomy: dict) -> str:
    """Keyword weak-labeller used for retrieval stratification and sampling."""
    t = text.lower()
    best_id = "thanks_or_other"
    best_hits = 0
    for intent in taxonomy["intents"]:
        hits = sum(1 for kw in intent.get("keywords", []) if kw in t)
        if hits > best_hits:
            best_hits = hits
            best_id = intent["id"]
    return best_id


def attach_weak_labels(pairs: pd.DataFrame) -> pd.DataFrame:
    import yaml

    from src.utils.paths import INTENTS_PATH

    with open(INTENTS_PATH, encoding="utf-8") as f:
        taxonomy = yaml.safe_load(f)
    pairs = pairs.copy()
    pairs["weak_intent"] = pairs["customer_text"].map(lambda x: weak_label_intent(x, taxonomy))
    return pairs


def _json_safe(rec: dict) -> dict:
    out = {}
    for k, v in rec.items():
        if pd.isna(v):
            out[k] = None
        elif hasattr(v, "isoformat"):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out


def save_processed(pairs: pd.DataFrame, name: str = "twitter_support_pairs") -> Path:
    ensure_dirs()
    out_parquet = PROCESSED_DIR / f"{name}.parquet"
    out_jsonl = PROCESSED_DIR / f"{name}.jsonl"
    out_csv = PROCESSED_DIR / f"{name}_sample.csv"
    pairs = pairs.copy()
    if "created_at" in pairs.columns:
        pairs["created_at"] = pairs["created_at"].astype(str)
    try:
        pairs.to_parquet(out_parquet, index=False)
    except Exception:
        # parquet optional if engine missing
        pass
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for rec in pairs.to_dict(orient="records"):
            f.write(json.dumps(_json_safe(rec), ensure_ascii=False) + "\n")
    pairs.head(200).to_csv(out_csv, index=False)
    print(f"Wrote {out_jsonl} ({len(pairs)} rows)")
    meta = {
        "brand": BRAND,
        "n_pairs": len(pairs),
        "weak_intent_counts": pairs["weak_intent"].value_counts().to_dict()
        if "weak_intent" in pairs.columns
        else {},
    }
    with open(PROCESSED_DIR / f"{name}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return out_jsonl


def load_pairs(path: Path | None = None) -> pd.DataFrame:
    path = path or (PROCESSED_DIR / "twitter_support_pairs.jsonl")
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python -m src.data.prepare --download --max-pairs 12000"
        )
    return pd.read_json(path, lines=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TwitterSupport pairs")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--csv", type=str, default=None, help="Path to twcs.csv")
    parser.add_argument("--max-pairs", type=int, default=12000)
    args = parser.parse_args()

    ensure_dirs()
    if args.csv:
        csv_path = Path(args.csv)
    elif args.download:
        csv_path = download_dataset()
    else:
        csv_path = RAW_DIR / "twcs.csv"
        if not csv_path.exists():
            csv_path = download_dataset()

    pairs = build_pairs(csv_path, max_pairs=args.max_pairs)
    pairs = attach_weak_labels(pairs)
    save_processed(pairs)


if __name__ == "__main__":
    main()
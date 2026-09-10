"""Embedding retrieval over historical TwitterSupport replies."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.paths import INDEX_DIR, PROCESSED_DIR, ensure_dirs

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class ReplyRetriever:
    def __init__(
        self,
        pairs: pd.DataFrame | None = None,
        index_dir: Path | None = None,
        model_name: str = MODEL_NAME,
    ) -> None:
        ensure_dirs()
        self.index_dir = index_dir or INDEX_DIR
        self.model_name = model_name
        self.model = None
        self.embeddings: np.ndarray | None = None
        self.meta: list[dict[str, Any]] = []
        if pairs is not None:
            self.build(pairs)
        else:
            self.load()

    def _get_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
        return self.model

    def build(self, pairs: pd.DataFrame) -> None:
        texts = pairs["customer_text"].astype(str).tolist()
        model = self._get_model()
        emb = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        self.embeddings = np.asarray(emb, dtype=np.float32)
        self.meta = pairs[
            ["pair_id", "customer_text", "brand_reply", "weak_intent"]
            if "weak_intent" in pairs.columns
            else ["pair_id", "customer_text", "brand_reply"]
        ].to_dict(orient="records")
        self.save()

    def save(self) -> None:
        assert self.embeddings is not None
        self.index_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.index_dir / "embeddings.npy", self.embeddings)
        with open(self.index_dir / "meta.jsonl", "w", encoding="utf-8") as f:
            for row in self.meta:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(self.index_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump({"model_name": self.model_name, "n": len(self.meta)}, f)

    def load(self) -> None:
        emb_path = self.index_dir / "embeddings.npy"
        meta_path = self.index_dir / "meta.jsonl"
        if not emb_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Retrieval index missing under {self.index_dir}. "
                "Run: python -m src.agent.retrieve --build"
            )
        self.embeddings = np.load(emb_path)
        self.meta = []
        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                self.meta.append(json.loads(line))
        cfg = self.index_dir / "config.json"
        if cfg.exists():
            self.model_name = json.loads(cfg.read_text(encoding="utf-8")).get(
                "model_name", self.model_name
            )

    def retrieve(self, query: str, top_k: int = 3, exclude_pair_id: str | None = None) -> list[dict[str, Any]]:
        assert self.embeddings is not None
        model = self._get_model()
        q = model.encode([query], normalize_embeddings=True)
        q = np.asarray(q, dtype=np.float32)[0]
        sims = self.embeddings @ q
        order = np.argsort(-sims)
        results = []
        for idx in order:
            item = dict(self.meta[int(idx)])
            if exclude_pair_id and item.get("pair_id") == exclude_pair_id:
                continue
            item["similarity"] = float(sims[int(idx)])
            results.append(item)
            if len(results) >= top_k:
                break
        return results


def build_index_from_processed(max_rows: int | None = 8000) -> ReplyRetriever:
    path = PROCESSED_DIR / "twitter_support_pairs.jsonl"
    pairs = pd.read_json(path, lines=True)
    if max_rows:
        # stratified-ish: sample per weak intent then fill
        if "weak_intent" in pairs.columns:
            parts = []
            intents = pairs["weak_intent"].unique()
            per = max(50, max_rows // max(1, len(intents)))
            for intent in intents:
                subset = pairs[pairs["weak_intent"] == intent]
                parts.append(subset.sample(n=min(len(subset), per), random_state=42))
            pairs = pd.concat(parts).drop_duplicates("pair_id")
            if len(pairs) > max_rows:
                pairs = pairs.sample(n=max_rows, random_state=42)
        else:
            pairs = pairs.head(max_rows)
    return ReplyRetriever(pairs=pairs.reset_index(drop=True))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--max-rows", type=int, default=8000)
    parser.add_argument("--query", type=str, default=None)
    args = parser.parse_args()
    if args.build:
        retriever = build_index_from_processed(max_rows=args.max_rows)
        print(f"Built index with {len(retriever.meta)} docs at {INDEX_DIR}")
    else:
        retriever = ReplyRetriever()
    if args.query:
        for hit in retriever.retrieve(args.query, top_k=3):
            print(f"[{hit['similarity']:.3f}] {hit['brand_reply'][:160]}")


if __name__ == "__main__":
    main()
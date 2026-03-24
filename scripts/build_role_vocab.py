from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from retrieval.query_rewriter import tokenize
from retrieval.query_rewriter import TECH_SKILLS, SOFT_SKILLS, ROLE_HINTS


CATALOG_STOPWORDS = {
    "test",
    "tests",
    "assessment",
    "assessments",
    "knowledge",
    "skills",
    "minutes",
    "minute",
    "duration",
    "remote",
    "adaptive",
    "professional",
    "individual",
    "contributor",
    "languages",
    "language",
    "job",
    "levels",
    "level",
    "description",
    "sample",
}


def build_role_vocab(
    catalog_path: str,
    top_k: int = 100,
    min_len: int = 4,
    min_df_ratio: float = 0.05,
    max_df_ratio: float = 0.4,
) -> dict:
    df = pd.read_json(catalog_path, lines=True) if catalog_path.endswith(".jsonl") else pd.read_parquet(catalog_path)
    texts = []
    if "doc_text" in df.columns:
        texts = df["doc_text"].astype(str).tolist()
    else:
        # fallback to name + description
        names = df.get("name", "").astype(str)
        desc = df.get("description", "").astype(str)
        texts = (names + " " + desc).tolist()

    doc_freq = Counter()
    N = len(texts)
    for txt in texts:
        seen = set()
        for tok in tokenize(txt):
            if len(tok) >= min_len and tok not in CATALOG_STOPWORDS:
                seen.add(tok)
        for tok in seen:
            doc_freq[tok] += 1

    min_df = max(1, int(N * min_df_ratio))
    max_df = max(1, int(N * max_df_ratio))
    filtered = [(tok, df) for tok, df in doc_freq.items() if min_df <= df <= max_df]
    # Sort by DF descending
    filtered.sort(key=lambda x: x[1], reverse=True)
    tokens = [tok for tok, _ in filtered[:top_k]]

    technical = [t for t in tokens if t in TECH_SKILLS]
    behavioral = [t for t in tokens if t in SOFT_SKILLS]
    roles = [t for t in tokens if t in ROLE_HINTS]
    generic = [t for t in tokens if t not in technical and t not in behavioral and t not in roles]

    return {
        "all": tokens,
        "technical": technical,
        "behavioral": behavioral,
        "roles": roles,
        "generic": generic,
        "meta": {
            "total_docs": N,
            "min_df": min_df,
            "max_df": max_df,
            "top_k": top_k,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Build a simple role/keyword vocab from catalog.")
    parser.add_argument("--catalog", required=True, help="Path to catalog JSONL/parquet with doc_text or name/description.")
    parser.add_argument("--out", default="data/catalog_role_vocab.json", help="Output JSON file.")
    parser.add_argument("--top-k", type=int, default=100, help="Top-k tokens to keep.")
    parser.add_argument("--min-len", type=int, default=4, help="Minimum token length.")
    parser.add_argument("--min-df-ratio", type=float, default=0.05, help="Min document frequency ratio to keep a token.")
    parser.add_argument("--max-df-ratio", type=float, default=0.4, help="Max document frequency ratio to keep a token.")
    args = parser.parse_args()

    vocab = build_role_vocab(
        args.catalog,
        top_k=args.top_k,
        min_len=args.min_len,
        min_df_ratio=args.min_df_ratio,
        max_df_ratio=args.max_df_ratio,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"Saved vocab to {args.out}")


if __name__ == "__main__":
    main()

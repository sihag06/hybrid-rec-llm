from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def summarize(df: pd.DataFrame) -> dict:
    out = {}
    out["total_queries"] = int(len(df))
    out["pos_in_bm25"] = int(df["pos_in_bm25"].sum())
    out["pos_in_vec"] = int(df["pos_in_vec"].sum())
    out["pos_in_hybrid"] = int(df["pos_in_hybrid"].sum())
    out["bm25_only"] = int(df["bm25_only"].sum())
    out["vec_only"] = int(df["vec_only"].sum())
    out["neither"] = int(df["neither"].sum())

    def rank_stats(col):
        s = df[col]
        found = s[s <= df["rank_bm25"].max()]  # filter out sentinel topn+1 values
        if len(found) == 0:
            return None
        return {
            "count": int(len(found)),
            "mean": float(found.mean()),
            "median": float(found.median()),
        }

    out["rank_bm25_stats"] = rank_stats("rank_bm25")
    out["rank_vec_stats"] = rank_stats("rank_vec")
    out["rank_hybrid_stats"] = rank_stats("rank_hybrid")
    return out


def main():
    parser = argparse.ArgumentParser(description="Summarize candidate_coverage.jsonl into compact JSON stats.")
    parser.add_argument("--input", default="runs/candidate_coverage.jsonl", help="Path to candidate_coverage.jsonl")
    parser.add_argument("--out", default="runs/candidate_coverage_stats.json", help="Path to write stats JSON")
    args = parser.parse_args()

    df = pd.read_json(args.input, lines=True)
    stats = summarize(df)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

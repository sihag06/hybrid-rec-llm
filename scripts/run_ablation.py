from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run ablation suite for retrieval settings.")
    parser.add_argument("--catalog", default="data/catalog_docs.jsonl")
    parser.add_argument("--train", default="data/Gen_AI Dataset.xlsx")
    parser.add_argument("--vector-index", default="data/faiss_index/index.faiss")
    parser.add_argument("--assessment-ids", default="data/embeddings/assessment_ids.json")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--topn-list", default="50,100,200,400", help="Comma-separated topn candidates to test")
    args = parser.parse_args()

    topn_vals = [int(x) for x in args.topn_list.split(",") if x.strip()]
    runs_dir = Path("runs/ablation")
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for topn in topn_vals:
        # BM25
        run_cmd(
            [
                sys.executable,
                "-m",
                "eval.run_eval",
                "--catalog",
                args.catalog,
                "--train",
                args.train,
                "--recommender",
                "bm25",
                "--topn-candidates",
                str(topn),
                "--out-dir",
                str(runs_dir / f"bm25_top{topn}"),
            ]
        )
        # Vector
        run_cmd(
            [
                sys.executable,
                "-m",
                "eval.run_eval",
                "--catalog",
                args.catalog,
                "--train",
                args.train,
                "--recommender",
                "vector",
                "--vector-index",
                args.vector_index,
                "--assessment-ids",
                args.assessment_ids,
                "--model",
                args.model,
                "--topn-candidates",
                str(topn),
                "--out-dir",
                str(runs_dir / f"vector_top{topn}"),
            ]
        )
        # Hybrid RRF
        run_cmd(
            [
                sys.executable,
                "-m",
                "eval.run_eval",
                "--catalog",
                args.catalog,
                "--train",
                args.train,
                "--recommender",
                "hybrid_rrf",
                "--vector-index",
                args.vector_index,
                "--assessment-ids",
                args.assessment_ids,
                "--model",
                args.model,
                "--topn-candidates",
                str(topn),
                "--rrf-k",
                "60",
                "--out-dir",
                str(runs_dir / f"hybrid_rrf_top{topn}"),
            ]
        )

        # Collect metrics
        for name in ["bm25", "vector", "hybrid_rrf"]:
            mpath = runs_dir / f"{name}_top{topn}" / "metrics.json"
            if mpath.exists():
                with open(mpath) as f:
                    metrics = json.load(f)
                summary.append(
                    {
                        "variant": f"{name}_top{topn}",
                        "train_recall@10": metrics["train"]["recall@10"],
                        "val_recall@10": metrics["val"]["recall@10"],
                        "train_mrr@10": metrics["train"]["mrr@10"],
                        "val_mrr@10": metrics["val"]["mrr@10"],
                    }
                )

    with open(runs_dir / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Ablation summary written to runs/ablation/ablation_summary.json")


if __name__ == "__main__":
    main()

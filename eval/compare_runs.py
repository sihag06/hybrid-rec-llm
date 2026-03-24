from __future__ import annotations

import json
import sys
from pathlib import Path


def load_metrics(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def compare(run_a: str, run_b: str) -> dict:
    m_a = load_metrics(Path(run_a) / "metrics.json")
    m_b = load_metrics(Path(run_b) / "metrics.json")
    def extract(m):
        return {
            "train_r10": m["train"]["recall@10"],
            "val_r10": m["val"]["recall@10"],
            "val_mrr10": m["val"]["mrr@10"],
        }
    return {"run_a": run_a, "run_b": run_b, "metrics_a": extract(m_a), "metrics_b": extract(m_b)}


def main():
    if len(sys.argv) != 3:
        print("Usage: python -m eval.compare_runs <run_dir_a> <run_dir_b>")
        sys.exit(1)
    result = compare(sys.argv[1], sys.argv[2])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

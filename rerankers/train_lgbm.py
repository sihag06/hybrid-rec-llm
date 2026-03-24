from __future__ import annotations

import argparse
import json
from pathlib import Path

from rerankers.lgbm_reranker import train_lgbm_ranker


def main():
    parser = argparse.ArgumentParser(description="Train LGBM reranker on listwise data.")
    parser.add_argument("--train", default="data/reranker/train_listwise_train.parquet")
    parser.add_argument("--val", default="data/reranker/train_listwise_val.parquet")
    parser.add_argument("--out-dir", default="models/reranker/v0.1.0")
    parser.add_argument(
        "--features",
        default="rank_bm25,rank_vec,rank_rrf,bm25_score,vec_score,rrf_score,duration_minutes,remote_support,adaptive_support,has_duration_constraint,duration_constraint_minutes",
        help="Comma-separated feature columns (exclude label/qid/query/assessment_id)",
    )
    args = parser.parse_args()

    feature_cols = [f.strip() for f in args.features.split(",") if f.strip()]
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    res = train_lgbm_ranker(args.train, args.val, args.out_dir, feature_cols)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

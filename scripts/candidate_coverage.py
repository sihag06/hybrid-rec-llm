from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd

from data.catalog_loader import load_catalog
from data.train_loader import load_train
from recommenders.bm25 import BM25Recommender
from recommenders.vector_recommender import VectorRecommender
from retrieval.vector_index import VectorIndex
from models.embedding_model import EmbeddingModel
from retrieval.query_rewriter import rewrite_query


def rank_in_list(ids: List[str], positives: set, topn: int) -> int:
    for i, aid in enumerate(ids, 1):
        if aid in positives:
            return i
    return topn + 1


def main():
    parser = argparse.ArgumentParser(description="Candidate coverage analysis for BM25 vs Vector vs Hybrid.")
    parser.add_argument("--catalog", default="data/catalog_docs_rich.jsonl")
    parser.add_argument("--train", required=True)
    parser.add_argument("--vector-index", required=True)
    parser.add_argument("--assessment-ids", required=True)
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--topn", type=int, default=200)
    parser.add_argument("--use-rewriter", action="store_true")
    parser.add_argument("--vocab", help="Optional vocab for rewriter.")
    parser.add_argument("--out", default="runs/candidate_coverage.jsonl")
    args = parser.parse_args()

    df_catalog, _, id_by_url = load_catalog(args.catalog)
    examples, label_report = load_train(args.train, id_by_url)

    bm25 = BM25Recommender(df_catalog)
    embed_model = EmbeddingModel(args.embedding_model)
    index = VectorIndex.load(args.vector_index)
    with open(args.assessment_ids) as f:
        ids = json.load(f)
    vec = VectorRecommender(embed_model, index, df_catalog, ids, k_candidates=args.topn)

    vocab = {}
    if args.use_rewriter and args.vocab:
        with open(args.vocab) as f:
            vocab = json.load(f)

    # Group by query string, union positives to avoid duplicate rows per query.
    grouped: Dict[str, set] = {}
    for ex in examples:
        grouped.setdefault(ex.query, set()).update(ex.relevant_ids)

    rows: List[Dict[str, Any]] = []
    topn = args.topn
    for raw_query, positives in grouped.items():
        q = raw_query
        if args.use_rewriter:
            rw = rewrite_query(q, catalog_vocab=vocab)
            q = rw.retrieval_query

        bm25_res = bm25.recommend(q, k=topn)
        vec_res = vec.recommend(q, k=topn)
        bm25_ids = [r if isinstance(r, str) else r["assessment_id"] for r in bm25_res]
        vec_ids = [r if isinstance(r, str) else r["assessment_id"] for r in vec_res]

        hybrid_ids = bm25_ids + vec_ids
        # simple union preserving order of appearance
        seen = set()
        hybrid_union = []
        for aid in hybrid_ids:
            if aid not in seen:
                hybrid_union.append(aid)
                seen.add(aid)

        rank_bm25 = rank_in_list(bm25_ids, positives, topn)
        rank_vec = rank_in_list(vec_ids, positives, topn)
        rank_hybrid = rank_in_list(hybrid_union[:topn], positives, topn)

        rows.append(
            {
                "query": raw_query,
                "rank_bm25": rank_bm25,
                "rank_vec": rank_vec,
                "rank_hybrid": rank_hybrid,
                "pos_in_bm25": rank_bm25 <= topn,
                "pos_in_vec": rank_vec <= topn,
                "pos_in_hybrid": rank_hybrid <= topn,
                "bm25_only": rank_bm25 <= topn and rank_vec > topn,
                "vec_only": rank_vec <= topn and rank_bm25 > topn,
                "neither": rank_bm25 > topn and rank_vec > topn,
                "positives": list(positives),
            }
        )

    Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_json(args.out, orient="records", lines=True)
    print(f"Saved candidate coverage to {args.out}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

from data.catalog_loader import load_catalog
from data.train_loader import load_train
from retrieval.query_rewriter import rewrite_query
from recommenders.bm25 import BM25Recommender
from recommenders.vector_recommender import VectorRecommender
from retrieval.vector_index import VectorIndex
from models.embedding_model import EmbeddingModel
from recommenders.hybrid_rrf import HybridRRFRecommender


def rank_of_first_positive(preds: List[str], positives: set, not_found: int) -> int:
    for i, p in enumerate(preds, 1):
        if p in positives:
            return i
    return not_found  # indicate not found within retrieved set


def main():
    parser = argparse.ArgumentParser(description="Evaluate impact of query rewriting on positive ranks.")
    parser.add_argument("--catalog", default="data/catalog_docs_rich.jsonl")
    parser.add_argument("--train", required=True, help="Train file (xlsx/jsonl) with labels")
    parser.add_argument("--vector-index", required=True, help="FAISS index path")
    parser.add_argument("--assessment-ids", required=True, help="assessment_ids.json aligned with index")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--topn", type=int, default=200, help="Candidates to fetch")
    parser.add_argument("--out", default="runs/rewrite_impact.jsonl")
    parser.add_argument("--vocab", help="Optional vocab json produced by build_role_vocab.py")
    args = parser.parse_args()

    df_catalog, _, id_by_url = load_catalog(args.catalog)
    examples, label_report = load_train(args.train, id_by_url)

    # Build recommenders
    bm25 = BM25Recommender(df_catalog)
    embed_model = EmbeddingModel(args.embedding_model)
    index = VectorIndex.load(args.vector_index)
    with open(args.assessment_ids) as f:
        ids = json.load(f)
    vec_rec = VectorRecommender(embed_model, index, df_catalog, ids, k_candidates=args.topn)
    hybrid = HybridRRFRecommender(bm25, vec_rec, topn_candidates=args.topn, rrf_k=60)

    vocab = {}
    if args.vocab:
        with open(args.vocab) as f:
            vocab = json.load(f)

    rows: List[Dict[str, Any]] = []

    not_found_val = args.topn + 1

    for ex in examples:
        positives = ex.relevant_ids

        # Raw query (no rewrite)
        raw_preds = hybrid.recommend(ex.query, k=200)
        raw_ids = [p["assessment_id"] if isinstance(p, dict) else p for p in raw_preds]
        raw_rank = rank_of_first_positive(raw_ids, positives, not_found=not_found_val)

        # Rule rewrite (no vocab)
        rw_rule = rewrite_query(ex.query)
        preds_rule = hybrid.recommend(rw_rule.retrieval_query, k=200)
        rule_ids = [p["assessment_id"] if isinstance(p, dict) else p for p in preds_rule]
        rule_rank = rank_of_first_positive(rule_ids, positives, not_found=not_found_val)

        # Rule + vocab rewrite
        rw_vocab = rewrite_query(ex.query, catalog_vocab=vocab)
        preds_vocab = hybrid.recommend(rw_vocab.retrieval_query, k=200)
        vocab_ids = [p["assessment_id"] if isinstance(p, dict) else p for p in preds_vocab]
        vocab_rank = rank_of_first_positive(vocab_ids, positives, not_found=not_found_val)

        rows.append(
            {
                "query": ex.query,
                "raw_rank": raw_rank,
                "rewrite_rank": rule_rank,
                "rewrite_vocab_rank": vocab_rank,
                "positives": list(positives),
            }
        )

    Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_json(args.out, orient="records", lines=True)
    print(f"Saved rewrite impact to {args.out}")


if __name__ == "__main__":
    main()

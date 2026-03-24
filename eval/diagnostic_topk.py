from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from data.catalog_loader import load_catalog
from data.train_loader import load_train
from recommenders.bm25 import BM25Recommender
from recommenders.vector_recommender import VectorRecommender
from recommenders.hybrid_rrf import HybridRRFRecommender, HybridRerankRecommender
from retrieval.vector_index import VectorIndex
from models.embedding_model import EmbeddingModel
from rerankers.cross_encoder import CrossEncoderReranker


def main():
    parser = argparse.ArgumentParser(description="Diagnostics: positives coverage in top-N candidates and top-10 rerank.")
    parser.add_argument("--catalog", default="data/catalog_docs.jsonl")
    parser.add_argument("--train", required=True)
    parser.add_argument("--vector-index", required=True)
    parser.add_argument("--assessment-ids", required=True)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--topn", type=int, default=200, help="Top-N candidates to inspect")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--output-dir", default="runs/diagnostic_topk")
    args = parser.parse_args()

    df_catalog, _, id_by_url = load_catalog(args.catalog)
    with open(args.assessment_ids) as f:
        ids = json.load(f)
    index = VectorIndex.load(args.vector_index)
    embed_model = EmbeddingModel(args.model)
    vector_rec = VectorRecommender(embed_model, index, df_catalog, ids, k_candidates=args.topn)
    bm25_rec = BM25Recommender(df_catalog)
    hybrid = HybridRRFRecommender(bm25_rec, vector_rec, topn_candidates=args.topn, rrf_k=args.rrf_k)
    reranker = CrossEncoderReranker(model_name=args.reranker_model)
    hybrid_rerank = HybridRerankRecommender(bm25_rec, vector_rec, reranker, df_catalog, topn_candidates=args.topn, rrf_k=args.rrf_k)

    examples, label_report = load_train(args.train, id_by_url)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir, "label_resolution_report.json").write_text(json.dumps(label_report, indent=2))

    rows = []
    coverage_fail = 0
    zero_topn = 0
    zero_top10 = 0
    for ex in examples:
        candidates = hybrid.recommend(ex.query, k=args.topn)
        reranked = hybrid_rerank.recommend(ex.query, k=10)
        pos_topn = len(set(candidates).intersection(ex.relevant_ids))
        pos_top10 = len(set(reranked).intersection(ex.relevant_ids))
        if pos_topn == 0:
            zero_topn += 1
        if pos_top10 == 0:
            zero_top10 += 1
        if pos_topn == 0:
            coverage_fail += 1
        rows.append(
            {
                "query": ex.query,
                "relevant_ids": list(ex.relevant_ids),
                "pos_in_topn": pos_topn,
                "pos_in_top10": pos_top10,
                "candidates": candidates,
                "reranked_top10": reranked,
            }
        )

    summary = {
        "total_queries": len(examples),
        "topn": args.topn,
        "zero_pos_in_topn": zero_topn,
        "zero_pos_in_top10": zero_top10,
        "coverage_failures": coverage_fail,
        "label_match_pct": label_report.get("matched_pct"),
    }
    with open(Path(args.output_dir) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(rows).to_json(Path(args.output_dir) / "per_query.jsonl", orient="records", lines=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

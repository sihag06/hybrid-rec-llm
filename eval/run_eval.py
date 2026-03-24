from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from data.catalog_loader import load_catalog
from data.train_loader import load_train, save_label_resolution_report
from eval.metrics import recall_at_k, mrr_at_k
from recommenders.dummy_random import DummyRandomRecommender
from recommenders.bm25 import BM25Recommender
from recommenders.vector_recommender import VectorRecommender
from recommenders.hybrid_rrf import HybridRRFRecommender, HybridRerankRecommender
from recommenders.hybrid_rrf_lgbm import HybridRRFLGBMRecommender
from retrieval.vector_index import VectorIndex
from models.embedding_model import EmbeddingModel
from rerankers.cross_encoder import CrossEncoderReranker
from rerankers.lgbm_reranker import LGBMReranker
from retrieval.query_rewriter import rewrite_query


def split_examples(examples, val_ratio=0.2, seed=42):
    import random

    rnd = random.Random(seed)
    shuffled = examples[:]
    rnd.shuffle(shuffled)
    cut = int(len(shuffled) * (1 - val_ratio))
    return shuffled[:cut], shuffled[cut:]


def run_eval(catalog_path: str, train_path: str, recommender_name: str, out_dir: str, seed: int = 42):
    df_catalog, catalog_by_id, id_by_url = load_catalog(catalog_path)
    examples, label_report = load_train(train_path, id_by_url)
    save_label_resolution_report(label_report, Path(out_dir) / "label_resolution_report.json")

    train_split, val_split = split_examples(examples, val_ratio=0.2, seed=seed)

    def make_recommender():
        if recommender_name == "dummy_random":
            return DummyRandomRecommender(df_catalog["assessment_id"].tolist(), seed=seed)
        if recommender_name == "bm25":
            return BM25Recommender(df_catalog)
        if recommender_name == "vector":
            # Expect doc_text present in df_catalog and provided index/ids/model via env/args; set below in main()
            raise RuntimeError("Vector recommender should be constructed in main with index and ids.")
        raise ValueError(f"Unknown recommender: {recommender_name}")

    recommender = make_recommender()

    def eval_split(split, split_name):
        preds_list: List[List[str]] = []
        gt_list: List[set] = []
        rows = []
        for ex in split:
            preds_raw = recommender.recommend(ex.query, k=10)
            preds = []
            for pr in preds_raw:
                if isinstance(pr, str):
                    preds.append(pr)
                elif isinstance(pr, dict) and "assessment_id" in pr:
                    preds.append(pr["assessment_id"])
            preds = preds[:10]
            preds_list.append(preds)
            gt_list.append(ex.relevant_ids)
            hits = len(set(preds).intersection(ex.relevant_ids))
            rows.append(
                {
                    "query": ex.query,
                    "relevant_ids": list(ex.relevant_ids),
                    "predicted_ids": preds,
                    "hits": hits,
                }
            )
        recall10 = sum(recall_at_k(g, p, 10) for g, p in zip(gt_list, preds_list)) / len(gt_list) if gt_list else 0.0
        recall5 = sum(recall_at_k(g, p, 5) for g, p in zip(gt_list, preds_list)) / len(gt_list) if gt_list else 0.0
        mrr10 = sum(mrr_at_k(g, p, 10) for g, p in zip(gt_list, preds_list)) / len(gt_list) if gt_list else 0.0
        return recall10, recall5, mrr10, rows

    train_r10, train_r5, train_mrr10, train_rows = eval_split(train_split, "train")
    val_r10, val_r5, val_mrr10, val_rows = eval_split(val_split, "val")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    metrics = {
        "recommender": recommender_name,
        "label_match_pct": label_report.get("matched_pct"),
        "train": {"recall@10": train_r10, "recall@5": train_r5, "mrr@10": train_mrr10, "n": len(train_split)},
        "val": {"recall@10": val_r10, "recall@5": val_r5, "mrr@10": val_mrr10, "n": len(val_split)},
    }
    with open(Path(out_dir) / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame(train_rows + val_rows).to_json(Path(out_dir) / "per_query_results.jsonl", orient="records", lines=True)
    worst = sorted(val_rows, key=lambda r: r["hits"])[:10]
    pd.DataFrame(worst).to_csv(Path(out_dir) / "worst_queries.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--train", required=True)
    parser.add_argument("--recommender", default="dummy_random")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vector-index", type=str, help="Path to FAISS index (for recommender=vector/hybrid_rrf)")
    parser.add_argument("--assessment-ids", type=str, help="Path to assessment_ids.json aligned with embeddings/index")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model for vector recommender")
    parser.add_argument("--topn-candidates", type=int, default=200, help="Top-N candidates to retrieve before fusion/rerank")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF smoothing constant")
    parser.add_argument("--reranker-model", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="Cross-encoder model for reranking")
    parser.add_argument("--lgbm-model", type=str, help="Path to trained LGBM model (for hybrid_rrf_lgbm)")
    parser.add_argument("--lgbm-features", type=str, help="Path to feature_schema.json for LGBM reranker")
    parser.add_argument("--use-rewriter", action="store_true", help="Rewrite queries before retrieval/rerank.")
    parser.add_argument("--vocab", type=str, help="Optional vocab JSON for rewriter boosts.")
    args = parser.parse_args()

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or f"runs/{run_id}_{args.recommender}"
    if args.recommender in {"vector", "hybrid_rrf", "hybrid_rrf_rerank", "hybrid_rrf_lgbm"}:
        if not args.vector_index or not args.assessment_ids:
            raise ValueError("Vector/hybrid recommender requires --vector-index and --assessment-ids")
        df_catalog, _, id_by_url = load_catalog(args.catalog)
        with open(args.assessment_ids) as f:
            ids = json.load(f)
        index = VectorIndex.load(args.vector_index)
        embed_model = EmbeddingModel(args.model)
        examples, label_report = load_train(args.train, id_by_url)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        save_label_resolution_report(label_report, Path(out_dir) / "label_resolution_report.json")
        vocab = {}
        if args.use_rewriter and args.vocab:
            with open(args.vocab) as f:
                vocab = json.load(f)

        train_split, val_split = split_examples(examples, val_ratio=0.2, seed=args.seed)
        vector_rec = VectorRecommender(embed_model, index, df_catalog, ids, k_candidates=args.topn_candidates)
        if args.recommender == "vector":
            recommender = vector_rec
        elif args.recommender == "hybrid_rrf":
            bm25_rec = BM25Recommender(df_catalog)
            recommender = HybridRRFRecommender(bm25_rec, vector_rec, topn_candidates=args.topn_candidates, rrf_k=args.rrf_k)
        elif args.recommender == "hybrid_rrf_rerank":
            bm25_rec = BM25Recommender(df_catalog)
            reranker = CrossEncoderReranker(model_name=args.reranker_model)
            recommender = HybridRerankRecommender(
                bm25_rec,
                vector_rec,
                reranker,
                df_catalog,
                topn_candidates=args.topn_candidates,
                rrf_k=args.rrf_k,
            )
        else:
            if not args.lgbm_model or not args.lgbm_features:
                raise ValueError("hybrid_rrf_lgbm requires --lgbm-model and --lgbm-features")
            bm25_rec = BM25Recommender(df_catalog)
            feature_cols = json.load(open(args.lgbm_features))
            if isinstance(feature_cols, dict) and "features" in feature_cols:
                feature_cols = feature_cols["features"]
            recommender = HybridRRFLGBMRecommender(
                bm25_rec,
                vector_rec,
                lgbm_model_path=args.lgbm_model,
                feature_cols=feature_cols,
                catalog_df=df_catalog,
                topn_candidates=args.topn_candidates,
                rrf_k=args.rrf_k,
            )

        def eval_split(split, split_name):
            preds_list = []
            gt_list = []
            rows = []
            for ex in split:
                retrieval_query = ex.query
                rerank_query = ex.query
                if args.use_rewriter:
                    rw = rewrite_query(ex.query, catalog_vocab=vocab)
                    retrieval_query = rw.retrieval_query
                    rerank_query = rw.rerank_query
                if args.recommender == "hybrid_rrf_rerank":
                    preds_raw = recommender.recommend(retrieval_query, k=10, rerank_query=rerank_query)
                else:
                    preds_raw = recommender.recommend(retrieval_query, k=10)
                preds = []
                for pr in preds_raw:
                    if isinstance(pr, str):
                        preds.append(pr)
                    elif isinstance(pr, dict) and "assessment_id" in pr:
                        preds.append(pr["assessment_id"])
                preds = preds[:10]
                preds_list.append(preds)
                gt_list.append(ex.relevant_ids)
                hits = len(set(preds).intersection(ex.relevant_ids))
                rows.append(
                    {
                        "query": ex.query,
                        "relevant_ids": list(ex.relevant_ids),
                        "predicted_ids": preds,
                        "hits": hits,
                    }
                )
            recall10 = sum(recall_at_k(g, p, 10) for g, p in zip(gt_list, preds_list)) / len(gt_list) if gt_list else 0.0
            recall5 = sum(recall_at_k(g, p, 5) for g, p in zip(gt_list, preds_list)) / len(gt_list) if gt_list else 0.0
            mrr10 = sum(mrr_at_k(g, p, 10) for g, p in zip(gt_list, preds_list)) / len(gt_list) if gt_list else 0.0
            return recall10, recall5, mrr10, rows

        train_r10, train_r5, train_mrr10, train_rows = eval_split(train_split, "train")
        val_r10, val_r5, val_mrr10, val_rows = eval_split(val_split, "val")
        metrics = {
            "recommender": args.recommender,
            "label_match_pct": label_report.get("matched_pct"),
            "train": {"recall@10": train_r10, "recall@5": train_r5, "mrr@10": train_mrr10, "n": len(train_split)},
            "val": {"recall@10": val_r10, "recall@5": val_r5, "mrr@10": val_mrr10, "n": len(val_split)},
            "config": {
                "topn_candidates": args.topn_candidates,
                "rrf_k": args.rrf_k,
                "model": args.model,
                "index": args.vector_index,
            },
        }
        with open(Path(out_dir) / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        pd.DataFrame(train_rows + val_rows).to_json(Path(out_dir) / "per_query_results.jsonl", orient="records", lines=True)
        worst = sorted(val_rows, key=lambda r: r["hits"])[:10]
        pd.DataFrame(worst).to_csv(Path(out_dir) / "worst_queries.csv", index=False)
        print(f"Run saved to {out_dir}")
    else:
        run_eval(args.catalog, args.train, args.recommender, out_dir, seed=args.seed)
        print(f"Run saved to {out_dir}")


if __name__ == "__main__":
    main()

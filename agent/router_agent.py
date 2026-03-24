from __future__ import annotations

"""
Router-style agent (minimal, deterministic) that orchestrates the tool stack:
- build_query_plan
- retrieve_candidates
- rerank_candidates
- apply_constraints
- explain

This is intentionally simple and does not require an LLM. You can swap
build_query_plan with an LLM-based planner that emits the same QueryPlan schema.
"""
import json
from typing import Callable

import pandas as pd

from data.catalog_loader import load_catalog
from recommenders.bm25 import BM25Recommender
from recommenders.vector_recommender import VectorRecommender
from retrieval.vector_index import VectorIndex
from models.embedding_model import EmbeddingModel
from rerankers.cross_encoder import CrossEncoderReranker

from tools.query_plan_tool import build_query_plan
from tools.retrieve_tool import retrieve_candidates
from tools.rerank_tool import rerank_candidates
from tools.constraints_tool import apply_constraints
from tools.explain_tool import explain


def load_resources():
    df_catalog, _, _ = load_catalog("data/catalog_docs_rich.jsonl")
    bm25 = BM25Recommender(df_catalog)
    embed = EmbeddingModel("BAAI/bge-small-en-v1.5")
    index = VectorIndex.load("data/faiss_index/index_bge.faiss")
    with open("data/embeddings_bge/assessment_ids.json") as f:
        ids = json.load(f)
    vec = VectorRecommender(embed, index, df_catalog, ids, k_candidates=200)
    return df_catalog, bm25, vec


def make_catalog_lookup(df_catalog: pd.DataFrame) -> Callable[[str], dict]:
    cat = df_catalog.set_index("assessment_id")

    def lookup(aid: str) -> dict:
        if aid in cat.index:
            return cat.loc[aid].to_dict()
        return {}

    return lookup


def route_query(user_text: str, vocab_path: str = "data/catalog_role_vocab.json") -> str:
    vocab = json.load(open(vocab_path)) if vocab_path else {}
    df_catalog, bm25, vec = load_resources()
    catalog_lookup = make_catalog_lookup(df_catalog)

    # 1) Plan (deterministic rewriter; swap with LLM-structured plan if desired)
    plan = build_query_plan(user_text, vocab=vocab)

    # 2) Clarification hook
    # Placeholder: in an interactive app, if plan.needs_clarification or coverage is weak,
    # ask a question and rebuild the plan with the user response.

    # 3) Retrieve (union of BM25 + vector)
    cand_set = retrieve_candidates(plan, bm25, vec, topn=200, catalog_df=df_catalog)

    # 4) Rerank (cross-encoder)
    reranker = CrossEncoderReranker(model_name="models/reranker_crossenc/v0.1.0")
    ranked = rerank_candidates(plan, cand_set, reranker, df_catalog, k=10)

    # 5) Apply constraints (stub; extend for duration/remote/adaptive)
    final_list = apply_constraints(plan, ranked)

    # 6) Explain
    summary = explain(plan, final_list, catalog_lookup)
    return summary


if __name__ == "__main__":
    import sys

    user_text = " ".join(sys.argv[1:]) or "Find a 1 hour culture fit assessment for a COO"
    print(route_query(user_text))

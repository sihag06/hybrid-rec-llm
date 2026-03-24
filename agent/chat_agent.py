from __future__ import annotations

"""
Chat-style agent using Gemini for planning + explanation, deterministic tools for retrieval/rerank.
Set GOOGLE_API_KEY in your environment.
"""
import json
import os
from typing import Callable

import pandas as pd

from data.catalog_loader import load_catalog
from recommenders.bm25 import BM25Recommender
from recommenders.vector_recommender import VectorRecommender
from retrieval.vector_index import VectorIndex
from models.embedding_model import EmbeddingModel
from rerankers.cross_encoder import CrossEncoderReranker

from tools.query_plan_tool_llm import build_query_plan_llm
from tools.query_plan_tool import build_query_plan as deterministic_plan
from tools.retrieve_tool import retrieve_candidates
from tools.rerank_tool import rerank_candidates
from tools.constraints_tool import apply_constraints
from tools.explain_tool import explain
from schemas.query_plan import QueryPlan


def load_resources():
    df_catalog, _, _ = load_catalog("data/catalog_docs_rich.jsonl")
    bm25 = BM25Recommender(df_catalog)
    embed = EmbeddingModel("BAAI/bge-small-en-v1.5")
    index = VectorIndex.load("data/faiss_index/index_bge.faiss")
    with open("data/embeddings_bge/assessment_ids.json") as f:
        ids = json.load(f)
    vec = VectorRecommender(embed, index, df_catalog, ids, k_candidates=200)
    catalog_by_id = {row["assessment_id"]: row for _, row in df_catalog.iterrows()}
    return df_catalog, bm25, vec, catalog_by_id


def make_catalog_lookup(df_catalog: pd.DataFrame) -> Callable[[str], dict]:
    cat = df_catalog.set_index("assessment_id")

    def lookup(aid: str) -> dict:
        if aid in cat.index:
            return cat.loc[aid].to_dict()
        return {}

    return lookup


def _maybe_clarify(plan: QueryPlan, cand_count: int, topn: int) -> str | None:
    # LLM-flagged clarification
    if plan.needs_clarification and plan.clarifying_question:
        return plan.clarifying_question
    # Coverage-based triggers
    if cand_count < max(10, int(0.25 * topn)):
        return "Results look thin. Clarify: are you looking for (1) personality/culture fit, (2) leadership judgment (SJT), or (3) role capability?"
    if plan.intent in {"BEHAVIORAL", "UNKNOWN", "MIXED"} and cand_count < max(20, int(0.5 * topn)):
        return "For culture/behavioral focus, choose: (1) personality/culture fit, (2) leadership judgment (SJT), or (3) role capability. Please pick one."
    return None


def run_chat(
    user_text: str,
    vocab_path: str = "data/catalog_role_vocab.json",
    model_name: str = "gemini-2.5-flash-lite",
    clarification_answer: str | None = None,
    topn: int = 200,
    verbose: bool = False,
):
    vocab = json.load(open(vocab_path)) if vocab_path and os.path.exists(vocab_path) else {}
    df_catalog, bm25, vec, catalog_by_id = load_resources()
    catalog_lookup = make_catalog_lookup(df_catalog)

    trace_id = f"trace-{abs(hash(user_text))}"
    log = {"trace_id": trace_id, "raw_query": user_text}

    # Plan with LLM; fallback deterministic if LLM fails
    try:
        plan = build_query_plan_llm(user_text, vocab=vocab, model_name=model_name)
        QueryPlan.model_validate(plan.dict())  # schema guard
        log["plan_source"] = "llm"
    except Exception as e:
        plan = deterministic_plan(user_text, vocab=vocab)
        log["plan_source"] = f"deterministic (llm_fail={str(e)})"
    log["query_plan"] = plan.dict()

    # Retrieve union
    cand_set = retrieve_candidates(plan, bm25, vec, topn=topn, catalog_df=df_catalog)
    if verbose:
        log["candidates"] = [c.model_dump() for c in cand_set.candidates[:10]]

    # Clarification loop
    question = _maybe_clarify(plan, cand_count=len(cand_set.candidates), topn=topn)
    if question and not clarification_answer:
        log["clarification"] = question
        if verbose:
            print(json.dumps(log, indent=2))
        return f"Clarification needed: {question}"
    if question and clarification_answer:
        clarified_text = f"{user_text}\nUser clarification: {clarification_answer}"
        try:
            plan = build_query_plan_llm(clarified_text, vocab=vocab, model_name=model_name)
            QueryPlan.model_validate(plan.dict())
        except Exception:
            plan = deterministic_plan(clarified_text, vocab=vocab)
        log["query_plan_clarified"] = plan.dict()
        cand_set = retrieve_candidates(plan, bm25, vec, topn=topn, catalog_df=df_catalog)
        if verbose:
            log["candidates_clarified"] = [c.model_dump() for c in cand_set.candidates[:10]]

    # Rerank
    reranker = CrossEncoderReranker(model_name="models/reranker_crossenc/v0.1.0")
    ranked = rerank_candidates(plan, cand_set, reranker, df_catalog, k=10)
    log["rerank"] = [item.model_dump() for item in ranked.items]

    # Constraints
    final_list = apply_constraints(plan, ranked, catalog_by_id, k=10)
    log["final"] = [item.model_dump() for item in final_list.items]

    # Explain
    summary = explain(plan, final_list, catalog_lookup)
    log["summary"] = summary

    # Compact output: top-10 with metadata
    final_results = []
    for item in final_list.items:
        meta = catalog_lookup(item.assessment_id)
        final_results.append(
            {
                "assessment_id": item.assessment_id,
                "score": item.score,
                "name": meta.get("name"),
                "url": meta.get("url"),
                "test_type_full": meta.get("test_type_full") or meta.get("test_type"),
                "duration": meta.get("duration_minutes") or meta.get("duration"),
            }
        )

    if verbose:
        log["final_results"] = final_results
        print(json.dumps(log, indent=2))
    else:
        print(json.dumps({"trace_id": trace_id, "final_results": final_results}, indent=2))

    return summary


if __name__ == "__main__":
    import sys

    if "GOOGLE_API_KEY" not in os.environ:
        print("Please set GOOGLE_API_KEY for Gemini.")
    user_text = " ".join(sys.argv[1:]) or "Find a 1 hour culture fit assessment for a COO"
    print(run_chat(user_text, verbose=False))

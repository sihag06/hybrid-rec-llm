from __future__ import annotations

"""
Minimal chat backend (FastAPI) that delegates to the agent app pipeline.

Run:
  uvicorn agent.server:app --reload --port 8000
"""

import uuid
import json
from typing import Optional, Callable
from collections import deque
import time
import math

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from functools import lru_cache
import os
from data.catalog_loader import load_catalog
from recommenders.bm25 import BM25Recommender
from recommenders.vector_recommender import VectorRecommender
from retrieval.vector_index import VectorIndex
from models.embedding_model import EmbeddingModel
from rerankers.cross_encoder import CrossEncoderReranker
from tools.query_plan_tool import build_query_plan
from tools.query_plan_tool_llm import build_query_plan_llm
from llm.nu_extract import NuExtractWrapper, default_query_rewrite_examples
from llm.qwen_rewriter import QwenRewriter
from tools.retrieve_tool import retrieve_candidates
from tools.rerank_tool import rerank_candidates
from tools.constraints_tool import apply_constraints


class ChatRequest(BaseModel):
    query: str
    clarification_answer: Optional[str] = None
    verbose: bool = False

class RecommendRequest(BaseModel):
    query: str
    llm_model: Optional[str] = None
    verbose: bool = False


def _make_catalog_lookup(df_catalog) -> Callable[[str], dict]:
    cat = df_catalog.set_index("assessment_id")
    def lookup(aid: str) -> dict:
        if aid in cat.index:
            return cat.loc[aid].to_dict()
        return {}
    return lookup


@lru_cache(maxsize=1)
def load_resources(llm_model_override: Optional[str] = None):
    df_catalog, _, _ = load_catalog("data/catalog_docs_rich.jsonl")
    bm25 = BM25Recommender(df_catalog)
    embed = EmbeddingModel("BAAI/bge-small-en-v1.5")
    index = VectorIndex.load("data/faiss_index/index_bge.faiss")
    with open("data/embeddings_bge/assessment_ids.json") as f:
        ids = json.load(f)
    vec = VectorRecommender(embed, index, df_catalog, ids, k_candidates=200)
    reranker = CrossEncoderReranker(model_name="models/reranker_crossenc/v0.1.0")
    lookup = _make_catalog_lookup(df_catalog)
    catalog_by_id = {row["assessment_id"]: row for _, row in df_catalog.iterrows()}
    vocab = {}
    vocab_path = "data/catalog_role_vocab.json"
    if os.path.exists(vocab_path):
        try:
            with open(vocab_path) as vf:
                vocab = json.load(vf)
        except Exception:
            vocab = {}
    # Optional LLM rewriter; choose via request override or env LLM_MODEL
    llm_extractor = None
    llm_model = llm_model_override or os.getenv("LLM_MODEL", "").strip()
    if not llm_model:
        llm_model = "Qwen/Qwen2.5-1.5B-Instruct"
    try:
        if llm_model.lower().startswith("qwen"):
            llm_extractor = QwenRewriter(model_name=llm_model, default_examples=default_query_rewrite_examples())
        elif not os.getenv("GOOGLE_API_KEY"):
            llm_extractor = NuExtractWrapper(default_examples=default_query_rewrite_examples())
    except Exception:
        llm_extractor = None
    return df_catalog, bm25, vec, reranker, lookup, vocab, llm_extractor, catalog_by_id


def _infer_remote_adaptive(meta: dict) -> (Optional[bool], Optional[bool]):
    remote = meta.get("remote_support", True if meta.get("remote_support") is None else meta.get("remote_support"))
    adaptive = meta.get("adaptive_support")
    text_blob = " ".join([str(meta.get("name", "")), str(meta.get("description", "")), str(meta.get("doc_text", ""))]).lower()
    if adaptive is None and "adaptive" in text_blob:
        adaptive = True
    return remote, adaptive


def _build_plan_with_fallback(query: str, vocab: dict, llm_extractor):
    """
    Build the query plan using the LLM rewriter (Qwen) when available, otherwise
    fall back to deterministic rewrite. No Gemini refinement to keep behavior predictable.
    """
    try:
        return build_query_plan(query, vocab=vocab, llm_extractor=llm_extractor)
    except Exception:
        return build_query_plan(query, vocab=vocab)


def _safe_num(val):
    try:
        if val is None:
            return None
        f = float(val)
        if math.isfinite(f):
            return f
    except Exception:
        return None
    return None


def _sanitize_debug(obj):
    """Recursively replace NaN/inf with None to keep JSON safe."""
    if isinstance(obj, dict):
        return {k: _sanitize_debug(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_debug(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_sanitize_debug(v) for v in obj)
    if isinstance(obj, (int, float)):
        return _safe_num(obj)
    return obj


CODE_TO_FULL = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


def _format_test_types(meta: dict) -> list[str]:
    if meta.get("test_type_full"):
        raw = meta["test_type_full"]
    elif meta.get("test_type"):
        raw = meta["test_type"]
    else:
        return []
    if isinstance(raw, list):
        vals = raw
    else:
        vals = str(raw).replace("/", ",").split(",")
    out = []
    for v in vals:
        v = v.strip()
        if not v:
            continue
        # Map letter codes to full names when applicable
        if len(v) == 1 and v in CODE_TO_FULL:
            out.append(CODE_TO_FULL[v])
        else:
            out.append(v)
    return out


def _run_pipeline(query: str, topn: int = 200, verbose: bool = False, llm_model: Optional[str] = None):
    if verbose:
        # For debugging, bypass cached resources to ensure fresh state
        load_resources.cache_clear()
    df_catalog, bm25, vec, reranker, lookup, vocab, llm_extractor, catalog_by_id = load_resources(llm_model_override=llm_model)
    plan = _build_plan_with_fallback(query, vocab=vocab, llm_extractor=llm_extractor)
    cand_set = retrieve_candidates(plan, bm25, vec, topn=topn, catalog_df=df_catalog)
    ranked = rerank_candidates(plan, cand_set, reranker, df_catalog, k=10)
    final_list = apply_constraints(plan, ranked, catalog_by_id, k=10)

    debug_payload = {}
    if verbose:
        debug_payload["plan"] = plan.dict()
        # If plan carries a source (from planner), include it
        if hasattr(plan, "plan_source"):
            debug_payload["plan_source"] = getattr(plan, "plan_source")
        # Capture NuExtract LLM debug if present
        if hasattr(plan, "llm_debug") and plan.llm_debug:
            debug_payload["llm_debug"] = plan.llm_debug
        if hasattr(cand_set, "fusion") and cand_set.fusion:
            debug_payload["fusion"] = cand_set.fusion
        debug_payload["candidates"] = [
            {
                "assessment_id": c.assessment_id,
                "bm25_rank": c.bm25_rank,
                "vector_rank": c.vector_rank,
                "hybrid_rank": c.hybrid_rank,
                "bm25_score": _safe_num(c.bm25_score),
                "vector_score": _safe_num(c.vector_score),
                "score": _safe_num(c.score),
            }
            for c in cand_set.candidates[: min(20, len(cand_set.candidates))]
        ]
        debug_payload["rerank"] = [
            {"assessment_id": r.assessment_id, "score": _safe_num(r.score)}
            for r in ranked.items[: min(20, len(ranked.items))]
        ]
        debug_payload["constraints"] = [
            {
                "assessment_id": r.assessment_id,
                "score": _safe_num(r.score),
                "debug": r.debug,
            }
            for r in final_list.items
        ]

    final_results = []
    for item in final_list.items:
        meta = lookup(item.assessment_id)
        remote, adaptive = _infer_remote_adaptive(meta)
        score = _safe_num(item.score)
        duration = _safe_num(meta.get("duration_minutes") or meta.get("duration"))
        duration_int = int(duration) if duration is not None else None
        description = meta.get("description") or meta.get("doc_text") or ""
        test_types = _format_test_types(meta)
        final_results.append(
            {
                "url": meta.get("url"),
                "name": meta.get("name"),
                "adaptive_support": "Yes" if adaptive else "No",
                "description": description,
                "duration": duration_int if duration_int is not None else 0,
                "remote_support": "Yes" if remote else "No",
                "test_type": test_types,
            }
        )
    # Guarantee at least one result if pipeline produced candidates
    if not final_results and ranked.items:
        item = ranked.items[0]
        meta = lookup(item.assessment_id)
        remote, adaptive = _infer_remote_adaptive(meta)
        duration = _safe_num(meta.get("duration_minutes") or meta.get("duration"))
        duration_int = int(duration) if duration is not None else 0
        final_results.append(
            {
                "url": meta.get("url"),
                "name": meta.get("name"),
                "adaptive_support": "Yes" if adaptive else "No",
                "description": meta.get("description") or meta.get("doc_text") or "",
                "duration": duration_int,
                "remote_support": "Yes" if remote else "No",
                "test_type": _format_test_types(meta),
            }
        )
    summary = {"plan": plan.intent, "top": len(final_results)}
    return final_results, summary, debug_payload


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # '*' cannot be used with credentials
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend assets
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Simple in-process rate limiter (max 5 requests per second)
_timestamps = deque()
_RATE_LIMIT = 5
_WINDOW = 1.0

def _allow_request() -> bool:
    now = time.time()
    while _timestamps and now - _timestamps[0] > _WINDOW:
        _timestamps.popleft()
    if len(_timestamps) < _RATE_LIMIT:
        _timestamps.append(now)
        return True
    return False


@app.post("/chat")
def chat(req: ChatRequest):
    if not _allow_request():
        return {"error": "rate limit exceeded"}
    trace_id = str(uuid.uuid4())
    final_results, summary, debug_payload = _run_pipeline(req.query, verbose=req.verbose)
    payload = {"trace_id": trace_id, "final_results": final_results}
    if req.verbose:
        payload["summary"] = summary
        payload["debug"] = _sanitize_debug(debug_payload)
    return payload


@app.post("/recommend")
def recommend(req: RecommendRequest):
    if not _allow_request():
        return {"error": "rate limit exceeded"}
    final_results, summary, debug_payload = _run_pipeline(req.query, verbose=req.verbose, llm_model=req.llm_model)
    resp = {"recommended_assessments": final_results}
    if req.verbose:
        resp["debug"] = _sanitize_debug(debug_payload)
        resp["summary"] = summary
    return resp


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    # Serve the SPA entry point
    return FileResponse("frontend/index.html")

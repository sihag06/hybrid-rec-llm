from __future__ import annotations

from typing import Dict, Optional

from retrieval.query_rewriter import rewrite_query
from schemas.query_plan import QueryPlan, DurationConstraint


def build_query_plan(raw_text: str, vocab: Optional[Dict] = None, llm_extractor=None) -> QueryPlan:
    """Plan builder using the rule-based rewriter; optional llm_extractor (e.g., NuExtract)."""
    rw = rewrite_query(raw_text, catalog_vocab=vocab, llm_extractor=llm_extractor)
    print("inside build query plan", rw)
    dur = None
    if rw.constraints and rw.constraints.duration:
        dur = DurationConstraint(mode=rw.constraints.duration.mode, minutes=rw.constraints.duration.minutes)
    language = rw.constraints.languages[0] if rw.constraints.languages else None
    plan_source = "deterministic"
    if rw.llm_debug:
        if rw.llm_debug.get("error"):
            plan_source = rw.llm_debug.get("model", "llm_error")
        else:
            plan_source = rw.llm_debug.get("model", "llm")
    
    print("plan source is", plan_source)

    plan = QueryPlan(
        intent=rw.intent,
        role_title=" ".join(rw.role_terms) if rw.role_terms else None,
        must_have_skills=rw.must_have_skills,
        soft_skills=rw.soft_skills,
        duration=dur,
        language=language,
        flags=rw.constraints.flags if rw.constraints else {},
        bm25_query=rw.retrieval_query,
        vec_query=rw.retrieval_query,
        rerank_query=rw.rerank_query,
        needs_clarification=False,
        clarifying_question=None,
        plan_source=plan_source,
        llm_debug=rw.llm_debug,
    )
    return plan

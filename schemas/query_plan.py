from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, validator


class DurationConstraint(BaseModel):
    mode: Optional[str] = Field(None, description="MAX or TARGET")
    minutes: Optional[int] = Field(None, description="Duration in minutes")


class QueryPlan(BaseModel):
    intent: str = Field(..., description="TECH | BEHAVIORAL | MIXED | UNKNOWN")
    role_title: Optional[str] = None
    must_have_skills: List[str] = Field(default_factory=list)
    soft_skills: List[str] = Field(default_factory=list)
    duration: Optional[DurationConstraint] = None
    language: Optional[str] = None
    flags: Optional[dict] = Field(default_factory=dict, description="Parsed flags: {'remote': Optional[bool], 'adaptive': Optional[bool]}")
    bm25_query: str = Field(..., description="Keyword-heavy query for BM25")
    vec_query: str = Field(..., description="Keyword-heavy query for vector retrieval")
    rerank_query: str = Field(..., description="Full context query for reranker")
    needs_clarification: bool = False
    clarifying_question: Optional[str] = None
    plan_source: Optional[str] = Field(default=None, description="Which planner produced this plan")
    llm_debug: Optional[dict] = Field(default=None, description="LLM debug info if available")

    @validator("intent")
    def intent_enum(cls, v):
        allowed = {"TECH", "BEHAVIORAL", "MIXED", "UNKNOWN"}
        if v not in allowed:
            raise ValueError(f"intent must be one of {allowed}")
        return v

    @validator("bm25_query", "vec_query", "rerank_query")
    def non_empty(cls, v):
        if not v or not str(v).strip():
            raise ValueError("query fields must be non-empty")
        return v.strip()

    @classmethod
    def json_schema(cls) -> dict:
        """Return JSON schema for structured LLM outputs / tool calling."""
        return cls.model_json_schema()

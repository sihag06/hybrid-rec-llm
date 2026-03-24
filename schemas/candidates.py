from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class Candidate(BaseModel):
    assessment_id: str
    source: Optional[str] = Field(None, description="bm25|vector|union|rerank")
    bm25_rank: Optional[int] = None
    vector_rank: Optional[int] = None
    hybrid_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    vector_score: Optional[float] = None
    score: Optional[float] = None  # generic slot (e.g., rerank)


class CandidateSet(BaseModel):
    candidates: List[Candidate]
    raw_bm25: List[str] = Field(default_factory=list)
    raw_vector: List[str] = Field(default_factory=list)
    fusion: Optional[dict] = None


class RankedItem(BaseModel):
    assessment_id: str
    score: float
    debug: Optional[dict] = None


class RankedList(BaseModel):
    items: List[RankedItem]

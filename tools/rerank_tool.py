from __future__ import annotations

from typing import List
import pandas as pd

from schemas.candidates import RankedItem, RankedList, CandidateSet


def rerank_candidates(plan, candidate_set: CandidateSet, reranker, catalog_df: pd.DataFrame, k: int = 10) -> RankedList:
    """Rerank candidate union using cross-encoder reranker."""
    catalog = catalog_df.set_index("assessment_id")
    scored: List[RankedItem] = []
    for cand in candidate_set.candidates:
        if cand.assessment_id not in catalog.index:
            continue
        doc_text = catalog.loc[cand.assessment_id].get("doc_text", "")
        score = reranker.score(plan.rerank_query, doc_text)
        scored.append(RankedItem(assessment_id=cand.assessment_id, score=float(score)))
    scored.sort(key=lambda x: x.score, reverse=True)
    return RankedList(items=scored[:k])

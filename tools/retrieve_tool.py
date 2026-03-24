from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from schemas.candidates import Candidate, CandidateSet


def retrieve_candidates(plan, bm25, vector, topn: int = 200, catalog_df: pd.DataFrame | None = None) -> CandidateSet:
    """Retrieve union of BM25 + vector using plan queries, then apply weighted RRF fusion."""
    bm25_res = bm25.recommend(plan.bm25_query, k=topn, return_scores=True)
    vec_res = vector.recommend(plan.vec_query, k=topn, return_scores=True)

    def to_id_scores(res):
        out_ids = []
        out_scores = {}
        for r in res:
            if isinstance(r, dict) and "assessment_id" in r:
                aid = r["assessment_id"]
                out_ids.append(aid)
                if "score" in r:
                    out_scores[aid] = r["score"]
            elif isinstance(r, (tuple, list)) and len(r) >= 2:
                aid, sc = r[0], r[1]
                out_ids.append(aid)
                try:
                    out_scores[aid] = float(sc)
                except Exception:
                    pass
            elif isinstance(r, str):
                out_ids.append(r)
        return out_ids, out_scores

    bm25_ids, bm25_scores = to_id_scores(bm25_res)
    vec_ids, vec_scores = to_id_scores(vec_res)

    bm25_pos = {aid: i + 1 for i, aid in enumerate(bm25_ids)}
    vec_pos = {aid: i + 1 for i, aid in enumerate(vec_ids)}

    # union preserving ids
    seen = set()
    union_ids: List[str] = []
    for aid in bm25_ids:
        if aid not in seen:
            union_ids.append(aid)
            seen.add(aid)
    for aid in vec_ids:
        if aid not in seen:
            union_ids.append(aid)
            seen.add(aid)

    # Query-adaptive weights + RRF fusion
    def _choose_fusion_weights(plan, raw_query: str) -> Tuple[float, float]:
        q = (plan.rerank_query or raw_query or "").strip()
        n_words = len(q.split())
        n_skills = len(plan.must_have_skills or [])
        n_soft = len(plan.soft_skills or [])
        w_b, w_v = 0.5, 0.5
        if n_skills >= 2:
            w_b += 0.2
            w_v -= 0.2
        if n_words >= 18:
            w_v += 0.2
            w_b -= 0.2
        if n_soft >= 2:
            w_v += 0.1
            w_b -= 0.1
        w_b = max(0.1, min(0.9, w_b))
        w_v = max(0.1, min(0.9, w_v))
        s = w_b + w_v
        return w_b / s, w_v / s

    w_bm25, w_vec = _choose_fusion_weights(plan, raw_query=plan.bm25_query)
    k_rrf = 60.0

    candidates: List[Candidate] = []
    scored: List[Tuple[float, Candidate]] = []
    for aid in union_ids:
        rb = bm25_pos.get(aid)
        rv = vec_pos.get(aid)
        rrf_b = w_bm25 / (k_rrf + rb) if rb is not None else 0.0
        rrf_v = w_vec / (k_rrf + rv) if rv is not None else 0.0
        fused = rrf_b + rrf_v
        cand = Candidate(
            assessment_id=aid,
            source="union",
            bm25_rank=rb,
            vector_rank=rv,
            hybrid_rank=None,  # filled after sorting by fused score
            bm25_score=bm25_scores.get(aid),
            vector_score=vec_scores.get(aid),
            score=fused,
        )
        scored.append((fused, cand))

    scored.sort(key=lambda x: x[0], reverse=True)
    for rank, (_, cand) in enumerate(scored[:topn]):
        cand.hybrid_rank = rank + 1
        candidates.append(cand)

    return CandidateSet(
        candidates=candidates,
        raw_bm25=bm25_ids,
        raw_vector=vec_ids,
        fusion={"w_bm25": w_bm25, "w_vec": w_vec, "k_rrf": k_rrf},
    )

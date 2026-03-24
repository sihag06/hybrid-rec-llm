from __future__ import annotations

import pandas as pd
from typing import Any, Dict, List

from recommenders.hybrid_rrf import HybridRRFRecommender
from rerankers.lgbm_reranker import LGBMReranker


class HybridRRFLGBMRecommender:
    """Hybrid retrieval (BM25 + vector via RRF) followed by LGBM reranking."""

    def __init__(
        self,
        bm25,
        vector,
        lgbm_model_path: str,
        feature_cols: List[str],
        catalog_df: pd.DataFrame,
        topn_candidates: int = 200,
        rrf_k: int = 60,
        rewriter=None,
        rewriter_vocab=None,
    ) -> None:
        self.hybrid = HybridRRFRecommender(
            bm25,
            vector,
            topn_candidates=topn_candidates,
            rrf_k=rrf_k,
            rewriter=rewriter,
            rewriter_vocab=rewriter_vocab,
        )
        self.reranker = LGBMReranker(lgbm_model_path, feature_cols)
        self.catalog = catalog_df.set_index("assessment_id")
        self.feature_cols = feature_cols

    def _build_features(self, query: str, candidates: List[str]) -> pd.DataFrame:
        # Minimal feature builder: requires columns already in catalog_df for static features.
        rows: List[Dict[str, Any]] = []
        for rank_rrf, aid in enumerate(candidates, start=1):
            row = {
                "assessment_id": aid,
                "rank_rrf": rank_rrf,
                # Placeholder scores; real implementation should carry bm25/vector/rrf scores if available.
                "bm25_score": 0.0,
                "vec_score": 0.0,
                "rrf_score": 1.0 / (60 + rank_rrf),
                "duration_minutes": self.catalog.loc[aid].get("duration_minutes") or self.catalog.loc[aid].get("duration"),
                "remote_support": self.catalog.loc[aid].get("remote_support"),
                "adaptive_support": self.catalog.loc[aid].get("adaptive_support"),
                "has_duration_constraint": 0,
                "duration_constraint_minutes": None,
            }
            row["rank_bm25"] = 1e6
            row["rank_vec"] = 1e6
            rows.append(row)
        df = pd.DataFrame(rows)
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        return df

    def recommend(self, query: str, k: int = 10) -> List[str]:
        candidates = self.hybrid.recommend(query, k=self.hybrid.topn_candidates)
        feats = self._build_features(query, candidates)
        reranked = self.reranker.rerank(feats, k=k)
        return reranked

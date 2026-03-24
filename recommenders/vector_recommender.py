from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from models.embedding_model import EmbeddingModel
from retrieval.vector_index import VectorIndex


class VectorRecommender:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_index: VectorIndex,
        catalog_df: pd.DataFrame,
        assessment_ids: List[str],
        k_candidates: int = 50,
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_index = vector_index
        self.catalog = catalog_df.set_index("assessment_id")
        self.assessment_ids = assessment_ids
        self.k_candidates = k_candidates

    def _preprocess_query(self, query: str) -> str:
        return re.sub(r"\s+", " ", query).strip()

    def recommend(self, query: str, k: int = 10, return_scores: bool = False) -> List[Dict[str, Any]]:
        query = self._preprocess_query(query)
        q_vec = self.embedding_model.encode([query], normalize=True, is_query=True)[0]
        scores, idx = self.vector_index.search(q_vec, k=self.k_candidates)
        results: List[Dict[str, Any]] = []
        for score, ix in zip(scores, idx):
            if ix < 0 or ix >= len(self.assessment_ids):
                continue
            aid = self.assessment_ids[ix]
            row = self.catalog.loc[aid].to_dict()
            rec = {"assessment_id": aid, "score": float(score), **row}
            if not return_scores:
                rec.pop("score", None)
            results.append(rec)
        return results[:k]

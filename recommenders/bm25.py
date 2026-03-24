from __future__ import annotations

import math
import re
from typing import List
from rank_bm25 import BM25Okapi

import pandas as pd

from recommenders.base import Recommender


def _tokenize(text: str) -> List[str]:
    # Regex tokenizer that keeps tech tokens like .net, c++, node.js, c#, entry-level.
    return re.findall(r"[a-z0-9]+(?:[.+#-][a-z0-9]+)*", text.lower())


def _join_field(field) -> str:
    if field is None:
        return ""
    if isinstance(field, (list, tuple)):
        return " ".join(str(x) for x in field if x)
    return str(field)


def _build_doc_text(row: pd.Series) -> str:
    parts = [
        row.get("name", ""),
        row.get("description", ""),
        row.get("test_type_full", ""),
        _join_field(row.get("job_levels")),
        _join_field(row.get("languages")),
    ]
    return " ".join(str(p) for p in parts if p)


class BM25Recommender(Recommender):
    """BM25 retriever using rank-bm25 (tokenization aligned with rewriter)."""

    def __init__(self, catalog_df: pd.DataFrame, k1: float = 1.5, b: float = 0.75) -> None:
        self.doc_ids: List[str] = catalog_df["assessment_id"].tolist()
        # Keep duration separately for post-filter/boost (do not inject into text).
        self.doc_duration: List[int | None] = []
        docs = [_build_doc_text(row) for _, row in catalog_df.iterrows()]
        self.tokens: List[List[str]] = []
        for (_, row), doc in zip(catalog_df.iterrows(), docs):
            self.tokens.append(_tokenize(doc))
            dur = row.get("duration_minutes") if "duration_minutes" in row else row.get("duration")
            try:
                if dur is None or (isinstance(dur, float) and math.isnan(dur)):
                    self.doc_duration.append(None)
                else:
                    self.doc_duration.append(int(round(float(dur))))
            except Exception:
                self.doc_duration.append(None)
        self.bm25 = BM25Okapi(self.tokens, k1=k1, b=b)

    def recommend(self, query: str, k: int = 10, return_scores: bool = False):
        q_tokens = _tokenize(query)
        max_minutes = self._parse_max_minutes(query)
        scores = self.bm25.get_scores(q_tokens)
        scored = []
        for idx, base_score in enumerate(scores):
            dur = self.doc_duration[idx]
            score = base_score
            if max_minutes is not None and dur is not None:
                if dur > max_minutes:
                    continue  # hard filter
                proximity = max(0.0, 1.0 - abs(dur - max_minutes) / (max_minutes + 1e-6))
                score = base_score * (1.0 + 0.1 * proximity)
            scored.append((score, self.doc_ids[idx]))
        # If filter removed everything, fall back to unfiltered scores
        if not scored:
            for idx, base_score in enumerate(scores):
                scored.append((base_score, self.doc_ids[idx]))
        scored.sort(key=lambda x: x[0], reverse=True)
        topk = scored[:k]
        if return_scores:
            return [{"assessment_id": doc_id, "score": float(score)} for score, doc_id in topk]
        return [doc_id for _, doc_id in topk]

    @staticmethod
    def _parse_max_minutes(query: str) -> float | None:
        """Extract a time budget from the query if present (e.g., '40 minutes', '30 min')."""
        m = re.search(r"(\d{1,3})\s*(?:minute|min)\b", query.lower())
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
        return None

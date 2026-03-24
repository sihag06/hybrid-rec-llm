from __future__ import annotations

import re
from typing import Optional

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Cross-encoder reranker for query-document scoring."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", max_length: int = 384) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.model = CrossEncoder(model_name, max_length=max_length)

    def _trim(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 2000:
            text = text[:2000]
        return text

    def score(self, query: str, doc: str) -> float:
        query = self._trim(query)
        doc = self._trim(doc)
        return float(self.model.predict([[query, doc]]))

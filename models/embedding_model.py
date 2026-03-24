from __future__ import annotations

from datetime import datetime
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Thin wrapper around SentenceTransformer with metadata."""

    def __init__(
        self,
        model_name: str,
        cache_dir: str = ".model_cache",
        device: str | None = None,
        query_prefix: Optional[str] = None,
        doc_prefix: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, cache_folder=cache_dir, device=device)
        # Sensible defaults for some popular retrieval models.
        lower_name = model_name.lower()
        if query_prefix is None and "bge" in lower_name:
            query_prefix = "Represent this query for retrieving relevant documents: "
        if doc_prefix is None and "bge" in lower_name:
            doc_prefix = "Represent this document for retrieval: "
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.metadata = {
            "model_name": model_name,
            "embedding_dim": self.model.get_sentence_embedding_dimension(),
            "max_seq_length": self.model.max_seq_length,
            "loaded_at": datetime.utcnow().isoformat(),
        }

    def encode(self, texts: List[str], normalize: bool = True, batch_size: int = 32, is_query: bool = False) -> np.ndarray:
        def add_prefix(t: str) -> str:
            if is_query and self.query_prefix:
                return f"{self.query_prefix}{t}"
            if not is_query and self.doc_prefix:
                return f"{self.doc_prefix}{t}"
            return t

        prefixed = [add_prefix(t or "") for t in texts]
        embeds = self.model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeds

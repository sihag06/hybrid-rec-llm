from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np


@dataclass
class VectorIndexConfig:
    index_type: str = "IndexFlatIP"
    embedding_dim: int = 384


class VectorIndex:
    def __init__(self, embeddings: np.ndarray, config: VectorIndexConfig | None = None):
        cfg = config or VectorIndexConfig(embedding_dim=embeddings.shape[1])
        if cfg.index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(cfg.embedding_dim)
        elif cfg.index_type == "IndexHNSWFlat":
            self.index = faiss.IndexHNSWFlat(cfg.embedding_dim, 32)
        else:
            raise ValueError(f"Unsupported index_type {cfg.index_type}")
        self.index.add(embeddings)
        self.config = cfg

    def search(self, query_vector: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        scores, idx = self.index.search(query_vector.astype(np.float32)[None, :], k)
        return scores[0], idx[0]

    def save(self, path: str, metadata: dict) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, path)
        meta = {
            **metadata,
            "index_type": self.config.index_type,
            "embedding_dim": self.config.embedding_dim,
            "saved_at": datetime.utcnow().isoformat(),
        }
        with open(Path(path).with_suffix(".json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str):
        index = faiss.read_index(path)
        obj = cls.__new__(cls)
        obj.index = index
        obj.config = VectorIndexConfig(index.d)
        return obj

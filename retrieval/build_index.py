from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from retrieval.vector_index import VectorIndex, VectorIndexConfig


def build_index(emb_path: str, ids_path: str, index_path: str, index_type: str = "IndexFlatIP") -> None:
    embeddings = np.load(emb_path)
    with open(ids_path) as f:
        assessment_ids = json.load(f)
    if embeddings.shape[0] != len(assessment_ids):
        raise ValueError(f"Embeddings count {embeddings.shape[0]} != ids count {len(assessment_ids)}")

    cfg = VectorIndexConfig(index_type=index_type, embedding_dim=embeddings.shape[1])
    index = VectorIndex(embeddings, cfg)
    metadata = {
        "num_vectors": len(assessment_ids),
        "embedding_dim": embeddings.shape[1],
        "index_type": index_type,
        "built_at": datetime.utcnow().isoformat(),
    }
    index.save(index_path, metadata)

    # Simple self-retrieval sanity check
    scores, idx = index.search(embeddings[0], k=1)
    if idx[0] != 0:
        raise RuntimeError("Self-retrieval sanity check failed (first vector not nearest to itself).")

    print(f"Index saved to {index_path} with {len(assessment_ids)} vectors and type {index_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and save FAISS index from embeddings.")
    parser.add_argument("--embeddings", default="data/embeddings/embeddings.npy")
    parser.add_argument("--ids", default="data/embeddings/assessment_ids.json")
    parser.add_argument("--index-path", default="data/faiss_index/index.faiss")
    parser.add_argument("--index-type", default="IndexFlatIP", choices=["IndexFlatIP", "IndexHNSWFlat"])
    args = parser.parse_args()

    Path(args.index_path).parent.mkdir(parents=True, exist_ok=True)
    build_index(args.embeddings, args.ids, args.index_path, index_type=args.index_type)

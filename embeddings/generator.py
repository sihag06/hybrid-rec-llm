from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from data.catalog_loader import make_assessment_id
from models.embedding_model import EmbeddingModel


def generate_embeddings(catalog_path: str, model_name: str, batch_size: int = 32, output_dir: str = "data/embeddings") -> Tuple[np.ndarray, List[str]]:
    df = pd.read_json(catalog_path, lines=True) if catalog_path.endswith(".jsonl") else pd.read_parquet(catalog_path)
    if "assessment_id" not in df.columns:
        if "url" in df.columns:
            df["assessment_id"] = df["url"].apply(make_assessment_id)
        else:
            raise KeyError("assessment_id not found and url missing to derive it.")
    df = df.sort_values("assessment_id")
    texts = df["doc_text"].tolist()
    ids = df["assessment_id"].tolist()

    model = EmbeddingModel(model_name)
    embeddings: List[np.ndarray] = []
    start = time.time()
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]
        embeds = model.encode(batch, normalize=True, batch_size=batch_size, is_query=False)
        embeddings.append(embeds)
    embeddings_arr = np.vstack(embeddings).astype(np.float32)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    np.save(Path(output_dir) / "embeddings.npy", embeddings_arr)
    with open(Path(output_dir) / "assessment_ids.json", "w") as f:
        json.dump(ids, f, indent=2)

    total_time = time.time() - start
    log = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_name": model_name,
        "num_documents": len(texts),
        "embedding_dim": embeddings_arr.shape[1],
        "batch_size": batch_size,
        "total_time_seconds": total_time,
        "avg_time_per_doc_ms": (total_time / len(texts) * 1000) if len(texts) else None,
        "normalized": True,
        "catalog_path": catalog_path,
    }
    with open(Path(output_dir) / "generation_log.json", "w") as f:
        json.dump(log, f, indent=2)
    return embeddings_arr, ids


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True, help="Enriched catalog with doc_text")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", default="data/embeddings")
    args = parser.parse_args()

    generate_embeddings(args.catalog, args.model, batch_size=args.batch_size, output_dir=args.output_dir)

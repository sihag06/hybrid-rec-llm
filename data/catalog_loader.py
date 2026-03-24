"""Catalog loader utilities: ID derivation, loading, and doc_text enrichment."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd


def make_assessment_id(url: str) -> str:
    """Derive a stable assessment_id slug from a URL."""
    slug = re.sub(r"https?://[^/]+", "", url)
    slug = slug.strip("/").replace("/", "_")
    slug = re.sub(r"[^a-z0-9_]", "", slug.lower())
    return slug or "unknown"


def _join(val) -> str:
    if val is None:
        return ""
    if isinstance(val, list):
        return " ".join(str(x) for x in val if x)
    return str(val)


def _build_doc_text(row: pd.Series) -> str:
    """Build a rich doc_text for BM25 + embedding retrieval."""
    parts = [
        row.get("name", ""),
        row.get("description", ""),
        _join(row.get("test_type_full") or row.get("test_type")),
        _join(row.get("job_levels")),
        _join(row.get("languages")),
    ]
    return " ".join(str(p) for p in parts if p).strip()


def load_catalog(
    path: str,
) -> Tuple[pd.DataFrame, Dict[str, dict], Dict[str, str]]:
    """Load catalog JSONL/Parquet; add assessment_id and doc_text.

    Returns:
        df_catalog  – full DataFrame
        catalog_by_id – dict[assessment_id -> row dict]
        id_by_url  – dict[url -> assessment_id]  (for label resolution)
    """
    if path.endswith(".jsonl"):
        df = pd.read_json(path, lines=True)
    else:
        df = pd.read_parquet(path)

    if "assessment_id" not in df.columns:
        df["assessment_id"] = df["url"].apply(make_assessment_id)

    if "doc_text" not in df.columns:
        df["doc_text"] = df.apply(_build_doc_text, axis=1)

    catalog_by_id: Dict[str, dict] = {}
    id_by_url: Dict[str, str] = {}
    for _, row in df.iterrows():
        aid = row["assessment_id"]
        catalog_by_id[aid] = row.to_dict()
        if "url" in row and row["url"]:
            id_by_url[row["url"]] = aid

    return df, catalog_by_id, id_by_url

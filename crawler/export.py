from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import structlog

from crawler.storage import Storage
from crawler.utils import now_iso

logger = structlog.get_logger(__name__)


def _normalize_row(row) -> dict:
    downloads = row["downloads"]
    if isinstance(downloads, str):
        try:
            downloads = json.loads(downloads)
        except Exception:
            downloads = None
    job_levels = row["job_levels"]
    if isinstance(job_levels, str):
        try:
            job_levels = json.loads(job_levels)
        except Exception:
            job_levels = [j.strip() for j in job_levels.split(",") if j.strip()]
    languages = row.get("languages")
    if isinstance(languages, str):
        try:
            languages = json.loads(languages)
        except Exception:
            languages = [l.strip() for l in languages.split(",") if l.strip()]
    duration_minutes = row["duration_minutes"]
    duration_hours = None
    if duration_minutes is not None:
        try:
            duration_hours = float(duration_minutes) / 60.0
        except Exception:
            duration_hours = None
    return {
        "url": row["url"],
        "name": row["name"],
        "description": row["description"],
        "test_type": row["test_type"],
        "test_type_full": row.get("test_type_full"),
        "remote_support": bool(row["remote_support"]) if row["remote_support"] is not None else None,
        "adaptive_support": bool(row["adaptive_support"]) if row["adaptive_support"] is not None else None,
        "duration": duration_minutes,
        "duration_hours": duration_hours,
        "job_levels": job_levels,
        "languages": languages,
        "downloads": downloads,
        "source": "shl_product_catalog",
        "crawled_at": now_iso(),
    }


def export_catalog(
    storage: Storage,
    parquet_path: str,
    jsonl_path: Optional[str] = None,
    min_count: int = 377,
    limit: Optional[int] = None,
) -> None:
    rows = storage.fetch_assessments()
    logger.info("export.assessments.fetched", count=len(rows))

    if len(rows) < min_count:
        raise RuntimeError(f"Validation failed: expected at least {min_count} assessments, got {len(rows)}")

    records = [_normalize_row(dict(r)) for r in rows]
    df = pd.DataFrame.from_records(records)
    if limit:
        df = df.head(limit)
        logger.info("export.limit.applied", limit=limit, rows=len(df))

    Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    logger.info("export.parquet.write", path=parquet_path, rows=len(df))

    if jsonl_path:
        df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)
        logger.info("export.jsonl.write", path=jsonl_path, rows=len(df))

    missing_desc = df["description"].isna().sum()
    missing_duration = df["duration"].isna().sum()
    logger.info(
        "export.summary",
        missing_description=missing_desc,
        missing_duration=missing_duration,
        test_type_counts=df["test_type"].value_counts(dropna=False).to_dict(),
    )

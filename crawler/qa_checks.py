from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def load_catalog(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Catalog file not found: {path}")
    if p.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if p.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported catalog format: {path}")


def qa_checks(df: pd.DataFrame) -> Dict[str, Any]:
    total = len(df)

    def pct_missing(col: str) -> float:
        return float(df[col].isna().mean()) * 100.0 if col in df else 100.0

    bool_sanity = {}
    for col in ["remote_support", "adaptive_support"]:
        if col in df:
            bool_sanity[col] = bool(
                df[col].dropna().apply(lambda x: isinstance(x, (bool, int))).all()
            )
        else:
            bool_sanity[col] = False

    description_lengths = df["description"].dropna().apply(lambda x: len(str(x))) if "description" in df else pd.Series(dtype=int)
    min_desc_len: Optional[int] = int(description_lengths.min()) if not description_lengths.empty else None

    return {
        "total": total,
        "count_gate": total >= 377,
        "missing_pct": {
            "description": pct_missing("description"),
            "test_type": pct_missing("test_type"),
            "remote_support": pct_missing("remote_support"),
            "adaptive_support": pct_missing("adaptive_support"),
            "duration_minutes": pct_missing("duration") if "duration" in df else pct_missing("duration_minutes"),
        },
        "url_uniqueness": {
            "unique_urls": int(df["url"].nunique()) if "url" in df else 0,
            "matches_row_count": bool("url" in df and df["url"].nunique() == total),
        },
        "description_quality": {
            "min_length": min_desc_len,
            "passed_min_30": bool(min_desc_len is not None and min_desc_len >= 30),
        },
        "test_type_distribution": df["test_type"].value_counts(dropna=False).to_dict() if "test_type" in df else {},
        "boolean_sanity": bool_sanity,
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python qa_checks.py <catalog.jsonl|catalog.parquet>")
        sys.exit(1)
    path = sys.argv[1]
    df = load_catalog(path)
    results = qa_checks(df)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

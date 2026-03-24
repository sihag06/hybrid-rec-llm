"""
Generate submission CSV by querying the recommend API for each test query.

Usage:
  python scripts/build_submission.py \
    --input data/dataset.xlsx \
    --sheet test \
    --output runs/submission.csv \
    --api-base https://agamp-llm-recommendation-backend.hf.space
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from typing import List, Optional

import pandas as pd
import requests


def load_queries(path: str, sheet: str) -> List[str]:
    df = pd.read_excel(path, sheet_name=sheet)
    query_col = None
    for col in df.columns:
        if str(col).strip().lower() == "query":
            query_col = col
            break
    if query_col is None:
        raise ValueError("No column named 'Query' found in the sheet.")
    queries = [str(q).strip() for q in df[query_col].dropna().tolist()]
    return queries


def fetch_urls(api_base: str, query: str, retries: int = 2, timeout: int = 60) -> List[str]:
    url = f"{api_base.rstrip('/')}/recommend"
    payload = {"query": query, "verbose": False}
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            recs = data.get("recommended_assessments") or []
            urls = []
            for rec in recs:
                u = rec.get("url") or rec.get("url_recommend") or ""
                if u:
                    urls.append(u)
                if len(urls) >= 10:
                    break
            return urls
        except Exception as e:
            if attempt >= retries:
                print(f"[warn] Query failed after retries: {query[:60]}... err={e}", file=sys.stderr)
                return []
            time.sleep(1)
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/dataset.xlsx", help="Path to the dataset Excel file")
    parser.add_argument("--sheet", default="Test-Set", help="Sheet name containing test queries")
    parser.add_argument("--output", default="runs/submission.csv", help="Where to write the submission CSV")
    parser.add_argument(
        "--api-base",
        default="http://localhost:8000",
        help="Base URL of the recommend API (e.g., https://<space>.hf.space)",
    )
    args = parser.parse_args()

    queries = load_queries(args.input, args.sheet)
    rows = []
    for q in queries:
        urls = fetch_urls(args.api_base, q)
        if not urls:
            rows.append({"Query": q, "Assessment_url": ""})
        else:
            for u in urls:
                rows.append({"Query": q, "Assessment_url": u})

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Query", "Assessment_url"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

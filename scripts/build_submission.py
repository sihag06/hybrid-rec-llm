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
import json
import os
import sys
import time
from typing import List, Optional

import pandas as pd
import requests
from tqdm import tqdm


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


def fetch_urls(api_base: str, query: str, retries: int = 2, timeout: int = 60) -> tuple[List[str], Optional[str], float]:
    start = time.time()
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
            return urls, None, time.time() - start
        except Exception as e:
            if attempt >= retries:
                print(f"[warn] Query failed after retries: {query[:60]}... err={e}", file=sys.stderr)
                return [], str(e), time.time() - start
            time.sleep(1)
    return [], "unknown_error", time.time() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/dataset.xlsx", help="Path to the dataset Excel file")
    parser.add_argument("--sheet", default="Test-Set", help="Sheet name containing test queries")
    parser.add_argument("--output", default="runs/submission.csv", help="Where to write the submission CSV")
    parser.add_argument("--progress-json", default=None, help="Optional JSONL file to append per-query progress")
    parser.add_argument(
        "--api-base",
        default="http://localhost:8000",
        help="Base URL of the recommend API (e.g., https://<space>.hf.space)",
    )
    parser.add_argument("--timeout", type=int, default=60, help="Per-request timeout in seconds")
    parser.add_argument("--retries", type=int, default=2, help="Retries per query")
    parser.add_argument("--batch-size", type=int, default=5, help="Sleep briefly after this many queries")
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between batches")
    parser.add_argument("--max-queries", type=int, default=10, help="Process only this many queries (default: 10)")
    args = parser.parse_args()

    queries = load_queries(args.input, args.sheet)
    if args.max_queries:
        queries = queries[: args.max_queries]

    csv_file = open(args.output, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=["Query", "Assessment_url"])
    writer.writeheader()

    rows = []
    for idx, q in enumerate(tqdm(queries, desc="Queries", unit="q"), 1):
        urls, err, elapsed = fetch_urls(args.api_base, q, retries=args.retries, timeout=args.timeout)
        if not urls:
            row = {"Query": q, "Assessment_url": ""}
            rows.append(row)
            writer.writerow(row)
        else:
            for u in urls:
                row = {"Query": q, "Assessment_url": u}
                rows.append(row)
                writer.writerow(row)

        csv_file.flush()
        os.fsync(csv_file.fileno())

        if args.progress_json:
            with open(args.progress_json, "a") as prog:
                prog.write(
                    json.dumps(
                        {
                            "query_index": idx,
                            "query": q,
                            "url_count": len(urls),
                            "urls": urls,
                            "elapsed_seconds": round(elapsed, 3),
                            "error": err,
                        }
                    )
                    + "\n"
                )

        if idx % args.batch_size == 0:
            time.sleep(args.sleep)

    csv_file.close()

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from pathlib import Path
from typing import List


def read_labels(path: str) -> List[str]:
    p = Path(path)
    if p.suffix == ".json":
        data = json.loads(p.read_text())
        if isinstance(data, dict) and "unmatched_labels" in data:
            return list(data["unmatched_labels"])
        if isinstance(data, list):
            return data
    if p.suffix in {".txt", ".csv"}:
        with p.open() as f:
            return [line.strip() for line in f if line.strip()]
    raise ValueError(f"Unsupported input format for labels: {path}")


def classify_html(status: int, html: str) -> str:
    if status == 404:
        return "NOT_FOUND"
    if status in {401, 403}:
        return "ACCESS_BLOCKED"
    lowered = html.lower()
    if any(marker in lowered for marker in ["page not found", "error occurred", "404"]):
        return "NOT_FOUND"
    markers = ["test type", "assessment length", "description", "catalogue__circle"]
    if any(m in lowered for m in markers):
        return "DETAIL_PAGE_VALID"
    return "NOT_CATALOG_ITEM"


def probe_url(url: str, timeout: int = 10) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
            status = resp.getcode()
            final_url = resp.geturl()
    except Exception as exc:  # pragma: no cover - network variability
        return {"url": url, "final_url": None, "status": None, "classification": "ERROR", "error": str(exc)}
    classification = classify_html(status, html)
    return {"url": url, "final_url": final_url, "status": status, "classification": classification, "error": None}


def main():
    parser = argparse.ArgumentParser(description="Probe unmatched label URLs and classify them.")
    parser.add_argument("--labels", required=True, help="Path to labels input (json with unmatched_labels, txt, or csv)")
    parser.add_argument("--output", required=True, help="CSV output path")
    args = parser.parse_args()

    labels = read_labels(args.labels)
    rows = []
    for url in labels:
        rows.append(probe_url(url))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["url", "final_url", "status", "classification", "error"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote probe results for {len(rows)} labels to {args.output}")


if __name__ == "__main__":
    sys.exit(main())

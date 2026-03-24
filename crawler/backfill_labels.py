from __future__ import annotations

import argparse
import asyncio
import csv
import os
from pathlib import Path

import structlog

from config import load_config
from crawler.fetcher import PlaywrightFetcher
from crawler.parser_detail import parse_detail_page
from crawler.robots import RobotsManager
from crawler.storage import PAGE_TYPE_DETAIL, PARSE_PARSED, PageRecord, Storage
from crawler.utils import RateLimiter

logger = structlog.get_logger(__name__)


async def backfill_from_probe(probe_csv: str, storage: Storage, fetcher: PlaywrightFetcher, robots: RobotsManager, allow_bypass: bool):
    with open(probe_csv) as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if row.get("classification") == "DETAIL_PAGE_VALID"]
    logger.info("backfill.labels.start", count=len(rows))
    for row in rows:
        url = row["url"]
        allowed = allow_bypass or robots.is_allowed(url)
        if not allowed:
            logger.warning("backfill.detail.disallowed", url=url)
            continue
        if allow_bypass:
            logger.warning("backfill.detail.disallowed.bypassed", url=url)
        result = await fetcher.fetch(url, page_type=PAGE_TYPE_DETAIL)
        storage.upsert_page(result.record)
        if result.error or not result.html:
            logger.error("backfill.detail.fetch_failed", url=url, error=result.error)
            continue
        parse_detail_page(result.html, url=url, storage=storage)
        storage.update_parse_status(url, PARSE_PARSED)


def main():
    parser = argparse.ArgumentParser(description="Backfill assessments from probed label URLs")
    parser.add_argument("--probe-csv", required=True, help="CSV from scripts/probe_unmatched_labels.py")
    parser.add_argument("--config", type=str, default=os.environ.get("CONFIG_PATH", "configs/config.yaml"))
    parser.add_argument("--sqlite", type=str, default="data/crawler.db")
    parser.add_argument("--allow-robots-bypass", action="store_true", help="Bypass robots.txt disallow (use responsibly)")
    args = parser.parse_args()

    config = load_config(args.config)
    rate_limiter = RateLimiter(
        base_delay=float(os.environ.get("REQUEST_DELAY_SECONDS", config.get("crawler", {}).get("request_delay_seconds", 1.5))),
        jitter=float(os.environ.get("JITTER_SECONDS", config.get("crawler", {}).get("jitter_seconds", 0.5))),
    )
    user_agent = os.environ.get("USER_AGENT", config.get("crawler", {}).get("user_agent"))
    max_retries = int(os.environ.get("MAX_RETRIES", config.get("crawler", {}).get("max_retries", 3)))

    storage = Storage(args.sqlite)
    robots = RobotsManager(robots_url="https://www.shl.com/robots.txt", user_agent=user_agent)
    robots.load()

    async def _runner():
        async with PlaywrightFetcher(user_agent=user_agent, rate_limiter=rate_limiter, max_retries=max_retries) as fetcher:
            await backfill_from_probe(args.probe_csv, storage, fetcher, robots, allow_bypass=args.allow_robots_bypass)

    asyncio.run(_runner())
    logger.info("backfill.labels.done")


if __name__ == "__main__":
    main()

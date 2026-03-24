from __future__ import annotations

import argparse
import asyncio
import os
from typing import Optional

import logging
import structlog

from config import load_config
from crawler.export import export_catalog
from crawler.fetcher import PlaywrightFetcher
from crawler.parser_catalog import parse_catalog_page
from crawler.parser_detail import parse_detail_page
from crawler.robots import RobotsManager
from crawler.storage import (
    PAGE_TYPE_CATALOG,
    PAGE_TYPE_DETAIL,
    PARSE_PENDING,
    Storage,
)
from crawler.utils import RateLimiter

logger = structlog.get_logger(__name__)


def configure_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, log_level.upper(), logging.INFO)),
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
    )


async def crawl_catalog(
    start_url: str,
    storage: Storage,
    fetcher: PlaywrightFetcher,
    robots: RobotsManager,
    max_discover: int | None = None,
    allow_robots_bypass: bool = False,
) -> None:
    frontier = [start_url]
    seen = set()
    total_discovered = 0

    while frontier:
        url = frontier.pop(0)
        if url in seen:
            continue
        seen.add(url)
        allowed = allow_robots_bypass or robots.is_allowed(url)
        if not allowed:
            logger.warning("catalog.fetch.disallowed", url=url)
            continue
        if allow_robots_bypass:
            logger.warning("catalog.fetch.disallowed.bypassed", url=url)
        result = await fetcher.fetch(url, page_type=PAGE_TYPE_CATALOG)
        storage.upsert_page(result.record)
        if result.error or not result.html:
            logger.error("catalog.fetch.failed", url=url, error=result.error)
            continue
        _, discovered_urls, next_pages = parse_catalog_page(result.html, source_url=url, storage=storage)
        total_discovered += len(discovered_urls)
        for next_url in next_pages:
            if next_url not in seen:
                frontier.append(next_url)
        if max_discover and total_discovered >= max_discover:
            logger.info("catalog.max_discover.reached", total=total_discovered, max=max_discover)
            break


async def crawl_details(
    storage: Storage,
    fetcher: PlaywrightFetcher,
    robots: RobotsManager,
    allow_robots_bypass: bool = False,
) -> None:
    pending = storage.get_pages_by_type(PAGE_TYPE_DETAIL, parse_status=PARSE_PENDING)
    logger.info("detail.queue", pending=len(pending))
    for page in pending:
        url = page["url"]
        allowed = allow_robots_bypass or robots.is_allowed(url)
        if not allowed:
            logger.warning("detail.fetch.disallowed", url=url)
            continue
        if allow_robots_bypass:
            logger.warning("detail.fetch.disallowed.bypassed", url=url)
        result = await fetcher.fetch(url, page_type=PAGE_TYPE_DETAIL)
        storage.upsert_page(result.record)
        if result.error or not result.html:
            logger.error("detail.fetch.failed", url=url, error=result.error)
            continue
        parse_detail_page(result.html, url=url, storage=storage)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Crawler pipeline")
    parser.add_argument("--mode", choices=["crawl_all", "discover", "details", "export"], default="crawl_all")
    parser.add_argument("--config", type=str, default=os.environ.get("CONFIG_PATH", "configs/config.yaml"))
    parser.add_argument("--parquet", type=str, default="data/catalog.parquet")
    parser.add_argument("--jsonl", type=str, default="data/catalog.jsonl")
    parser.add_argument(
        "--max-discover",
        type=int,
        default=None,
        help="Limit number of detail URLs discovered (for smoke tests)",
    )
    parser.add_argument(
        "--limit-export",
        type=int,
        default=None,
        help="Limit number of rows exported (for smoke tests)",
    )
    parser.add_argument(
        "--allow-robots-bypass",
        action="store_true",
        help="Bypass robots.txt disallow (for testing; use responsibly)",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    configure_logging(config.get("app", {}).get("log_level", "INFO"))
    crawler_cfg = config.get("crawler", {})
    rate_limiter = RateLimiter(
        base_delay=float(os.environ.get("REQUEST_DELAY_SECONDS", crawler_cfg.get("request_delay_seconds", 1.5))),
        jitter=float(os.environ.get("JITTER_SECONDS", crawler_cfg.get("jitter_seconds", 0.5))),
    )
    user_agent = os.environ.get("USER_AGENT", crawler_cfg.get("user_agent"))
    start_url = os.environ.get("START_URL", crawler_cfg.get("start_url"))
    max_retries = int(os.environ.get("MAX_RETRIES", crawler_cfg.get("max_retries", 3)))
    sqlite_path = crawler_cfg.get("sqlite_path", "data/crawler.db")
    allow_bypass = args.allow_robots_bypass or os.environ.get("ALLOW_ROBOTS_BYPASS", "").lower() in {"1", "true", "yes"}

    storage = Storage(sqlite_path)
    robots = RobotsManager(robots_url="https://www.shl.com/robots.txt", user_agent=user_agent)
    robots.load()

    async def _runner():
        async with PlaywrightFetcher(user_agent=user_agent, rate_limiter=rate_limiter, max_retries=max_retries) as fetcher:
            if args.mode in {"crawl_all", "discover"}:
                await crawl_catalog(start_url, storage, fetcher, robots, max_discover=args.max_discover, allow_robots_bypass=allow_bypass)
            if args.mode in {"crawl_all", "details"}:
                await crawl_details(storage, fetcher, robots, allow_robots_bypass=allow_bypass)

    if args.mode in {"crawl_all", "discover", "details"}:
        asyncio.run(_runner())

    if args.mode == "export":
        export_catalog(
            storage,
            parquet_path=args.parquet,
            jsonl_path=args.jsonl,
            limit=args.limit_export,
            min_count=1 if args.limit_export else 377,
        )


if __name__ == "__main__":
    main()

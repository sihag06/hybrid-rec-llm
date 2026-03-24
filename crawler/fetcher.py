from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

import structlog
from playwright.async_api import async_playwright, Browser, Page

from crawler.storage import PageRecord
from crawler.utils import RateLimiter

logger = structlog.get_logger(__name__)


@dataclass
class FetchResult:
    record: PageRecord
    status: Optional[int]
    html: Optional[str]
    error: Optional[str]


class PlaywrightFetcher:
    """Thin wrapper around Playwright with polite rate limiting."""

    def __init__(
        self,
        user_agent: str,
        rate_limiter: RateLimiter,
        max_retries: int = 3,
    ) -> None:
        self.user_agent = user_agent
        self.rate_limiter = rate_limiter
        self.max_retries = max_retries
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._page: Optional[Page] = None

    async def __aenter__(self) -> "PlaywrightFetcher":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def start(self) -> None:
        if self._page:
            return
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=True)
        context = await self._browser.new_context(user_agent=self.user_agent)
        self._page = await context.new_page()
        logger.info("fetcher.started", user_agent=self.user_agent)

    async def close(self) -> None:
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self._browser = None
        self._page = None
        logger.info("fetcher.closed")

    async def fetch(self, url: str, page_type: str) -> FetchResult:
        assert self._page, "Fetcher must be started before fetch()"
        attempt = 0
        last_error: Optional[str] = None
        html: Optional[str] = None
        status: Optional[int] = None

        while attempt < self.max_retries:
            attempt += 1
            self.rate_limiter.sleep()
            logger.info("fetcher.request", url=url, attempt=attempt)
            try:
                response = await self._page.goto(url, wait_until="networkidle", timeout=20000)
                status = response.status if response else None
                html = await self._page.content()
                return FetchResult(
                    record=PageRecord(url=url, page_type=page_type, http_status=status, html=html),
                    status=status,
                    html=html,
                    error=None,
                )
            except Exception as exc:  # pragma: no cover - network variability
                last_error = str(exc)
                logger.warning("fetcher.request.error", url=url, error=last_error, attempt=attempt)
        return FetchResult(
            record=PageRecord(url=url, page_type=page_type, http_status=status, html=html, error=last_error),
            status=status,
            html=html,
            error=last_error,
        )


def fetch_sync(url: str, page_type: str, user_agent: str, rate_limiter: RateLimiter, max_retries: int = 3) -> FetchResult:
    async def _runner():
        async with PlaywrightFetcher(user_agent=user_agent, rate_limiter=rate_limiter, max_retries=max_retries) as fetcher:
            return await fetcher.fetch(url, page_type)

    return asyncio.run(_runner())

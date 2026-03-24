from __future__ import annotations

import hashlib
import random
import time
import urllib.parse
from datetime import datetime, timezone
from typing import Iterable


def canonicalize_url(url: str) -> str:
    """Normalize URL by stripping fragments/query trackers and trailing slashes."""
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    filtered_query = [(k, v) for k, v in query if not k.lower().startswith("utm_")]
    cleaned_query = urllib.parse.urlencode(filtered_query, doseq=True)
    path = parsed.path if parsed.path != "/" else ""
    # Keep trailing slash for non-root paths to avoid 404s on detail pages.
    if path and not path.endswith("/"):
        path = path
    normalized = parsed._replace(query=cleaned_query, fragment="", path=path).geturl()
    return normalized or url


def make_assessment_id(url: str) -> str:
    """Deterministic ID from canonical URL."""
    canonical = canonicalize_url(url)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RateLimiter:
    """Coarse rate limiter with jitter to respect polite crawling."""

    def __init__(self, base_delay: float, jitter: float) -> None:
        self.base_delay = base_delay
        self.jitter = jitter
        self._last_ts = 0.0

    def sleep(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_ts
        delay = self.base_delay + random.uniform(0, self.jitter)
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_ts = time.monotonic()


def batched(iterable: Iterable, size: int):
    """Yield fixed-size batches from an iterable."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch

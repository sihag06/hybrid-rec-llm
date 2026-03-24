from __future__ import annotations

from typing import List, Tuple
from urllib.parse import urljoin

import structlog
from bs4 import BeautifulSoup

from crawler.storage import (
    PAGE_TYPE_DETAIL,
    PARSE_PARSED,
    PageRecord,
    Storage,
)
from crawler.utils import canonicalize_url, now_iso

logger = structlog.get_logger(__name__)

ALLOWED_TEST_TYPES = {"A", "B", "C", "D", "E", "K", "P", "S"}
GREEN_TOKENS = ["green", "#8ac640", "rgb(138", "rgb(103", "0, 167, 83", "8ac640"]


def _has_green_indicator(cell) -> bool:
    for el in cell.find_all(True):
        style = (el.get("style") or "").lower()
        classes = " ".join(el.get("class", [])).lower() if isinstance(el.get("class"), list) else str(el.get("class") or "").lower()
        combined = f"{style} {classes}"
        if any(tok in combined for tok in GREEN_TOKENS):
            return True
        if "-yes" in classes or "catalogue__circle" in classes:
            return True
        fill = (el.get("fill") or "").lower()
        if any(tok in fill for tok in GREEN_TOKENS):
            return True
        # Generic icon/dot detection (when color is applied via CSS)
        if el.name in {"svg", "circle", "path", "i"}:
            return True
        if "dot" in classes or "indicator" in classes:
            return True
    return False


def extract_catalog_entries(html: str) -> List[dict]:
    """Parse catalog page for individual test solutions.

    This is intentionally defensive; selectors may change on shl.com. We look for anchors within
    sections that mention "Individual Test Solutions" or tables with product rows.
    """
    soup = BeautifulSoup(html, "lxml")
    entries = []

    tables = soup.find_all("table")
    for table in tables:
        headers = " ".join(th.get_text(" ", strip=True) for th in table.find_all("th"))
        if "Individual Test Solutions" not in headers and "Assessment" not in headers:
            continue
        for row in table.find_all("tr"):
            link = row.find("a", href=True)
            if not link:
                continue
            name = link.get_text(strip=True)
            detail_url = link["href"]
            badges_text = [span.get_text("", strip=True) for span in row.find_all("span")]
            test_letters = []
            for token in badges_text:
                token = token.strip()
                if len(token) == 1 and token in ALLOWED_TEST_TYPES:
                    test_letters.append(token)
            test_type = ",".join(dict.fromkeys(test_letters)) or None
            tds = row.find_all("td")
            remote = None
            adaptive = None
            if len(tds) >= 3:
                remote = _has_green_indicator(tds[1])
                adaptive = _has_green_indicator(tds[2])
            else:
                flat_badges = " ".join(badges_text).lower()
                remote = "remote" in flat_badges
                adaptive = "adaptive" in flat_badges or "irt" in flat_badges
            entries.append(
                {
                    "name": name,
                    "url": detail_url,
                    "test_type": test_type or None,
                    "remote_support": remote if remote else None,
                    "adaptive_support": adaptive if adaptive else None,
                }
            )
    return entries


def find_next_pages(html: str, source_url: str) -> List[str]:
    """Find pagination links (Next or numbered) and resolve to absolute URLs."""
    soup = BeautifulSoup(html, "lxml")
    urls = []
    for link in soup.find_all("a", href=True):
        text = link.get_text(" ", strip=True).lower()
        if "next" in text or text.isdigit():
            urls.append(canonicalize_url(urljoin(source_url, link["href"])))
    # de-duplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def parse_catalog_page(html: str, source_url: str, storage: Storage) -> Tuple[int, List[str], List[str]]:
    entries = extract_catalog_entries(html)
    discovered_urls: List[str] = []

    for entry in entries:
        detail_url = canonicalize_url(urljoin(source_url, entry["url"]))
        discovered_urls.append(detail_url)
        storage.upsert_page(
            PageRecord(
                url=detail_url,
                page_type=PAGE_TYPE_DETAIL,
            )
        )
        storage.upsert_assessment(
            {
                "url": detail_url,
                "name": entry.get("name"),
                "test_type": entry.get("test_type"),
                "remote_support": entry.get("remote_support"),
                "adaptive_support": entry.get("adaptive_support"),
                "source_catalog_page": canonicalize_url(source_url),
                "discovered_at": now_iso(),
            }
        )

    storage.update_parse_status(source_url, PARSE_PARSED)
    next_pages = find_next_pages(html, source_url)
    logger.info(
        "catalog.parse.summary",
        source_url=source_url,
        discovered=len(discovered_urls),
        next_pages=len(next_pages),
    )
    return len(entries), discovered_urls, next_pages

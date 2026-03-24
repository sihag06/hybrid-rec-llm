from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import structlog

from crawler.utils import canonicalize_url, make_assessment_id, now_iso

logger = structlog.get_logger(__name__)


PAGE_TYPE_CATALOG = "CATALOG"
PAGE_TYPE_DETAIL = "DETAIL"

PARSE_PENDING = "PENDING"
PARSE_PARSED = "PARSED"
PARSE_FAILED = "FAILED"


@dataclass
class PageRecord:
    url: str
    page_type: str
    http_status: Optional[int] = None
    html: Optional[str] = None
    error: Optional[str] = None
    retry_count: int = 0
    parse_status: str = PARSE_PENDING


class Storage:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.ensure_schema()

    def ensure_schema(self) -> None:
        logger.info("storage.schema.ensure", db_path=self.db_path)
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS pages (
                url TEXT PRIMARY KEY,
                url_canonical TEXT UNIQUE,
                page_type TEXT,
                http_status INTEGER,
                fetched_at TEXT,
                html TEXT,
                error TEXT,
                retry_count INTEGER DEFAULT 0,
                parse_status TEXT DEFAULT 'PENDING'
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS assessments (
                assessment_id TEXT PRIMARY KEY,
                url TEXT UNIQUE,
                name TEXT,
                description TEXT,
                test_type TEXT,
                test_type_full TEXT,
                remote_support INTEGER,
                adaptive_support INTEGER,
                duration_minutes INTEGER,
                job_levels TEXT,
                languages TEXT,
                downloads TEXT,
                source_catalog_page TEXT,
                discovered_at TEXT,
                last_updated_at TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS crawl_meta (
                run_id TEXT,
                started_at TEXT,
                finished_at TEXT,
                total_catalog_pages INTEGER,
                total_detail_pages INTEGER,
                individual_assessment_count INTEGER,
                notes TEXT
            )
            """
        )
        self.conn.commit()

    def upsert_page(self, record: PageRecord) -> None:
        canonical = canonicalize_url(record.url)
        logger.debug("storage.page.upsert", url=record.url, page_type=record.page_type)
        self.conn.execute(
            """
            INSERT INTO pages (url, url_canonical, page_type, http_status, fetched_at, html, error, retry_count, parse_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                page_type=excluded.page_type,
                http_status=excluded.http_status,
                fetched_at=excluded.fetched_at,
                html=excluded.html,
                error=excluded.error,
                retry_count=excluded.retry_count,
                parse_status=excluded.parse_status
            """,
            (
                record.url,
                canonical,
                record.page_type,
                record.http_status,
                now_iso(),
                record.html,
                record.error,
                record.retry_count,
                record.parse_status,
            ),
        )
        self.conn.commit()

    def update_parse_status(self, url: str, status: str) -> None:
        self.conn.execute("UPDATE pages SET parse_status=? WHERE url=?", (status, url))
        self.conn.commit()

    def get_pages_by_type(self, page_type: str, parse_status: Optional[str] = None) -> List[sqlite3.Row]:
        cur = self.conn.cursor()
        if parse_status:
            cur.execute(
                "SELECT * FROM pages WHERE page_type=? AND parse_status=? ORDER BY url", (page_type, parse_status)
            )
        else:
            cur.execute("SELECT * FROM pages WHERE page_type=? ORDER BY url", (page_type,))
        return cur.fetchall()

    def upsert_assessment(self, data: Dict[str, Any]) -> None:
        url = data["url"]
        assessment_id = data.get("assessment_id") or make_assessment_id(url)
        data = {**data, "assessment_id": assessment_id}
        downloads = data.get("downloads")
        if downloads is not None and not isinstance(downloads, str):
            downloads = json.dumps(downloads)
        job_levels = data.get("job_levels")
        if isinstance(job_levels, (list, tuple)):
            job_levels = json.dumps(job_levels)
        languages = data.get("languages")
        if isinstance(languages, (list, tuple)):
            languages = json.dumps(languages)

        logger.debug("storage.assessment.upsert", url=url)
        self.conn.execute(
            """
            INSERT INTO assessments (
                assessment_id, url, name, description, test_type, test_type_full, remote_support, adaptive_support,
                duration_minutes, job_levels, languages, downloads, source_catalog_page, discovered_at, last_updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(assessment_id) DO UPDATE SET
                url=excluded.url,
                name=COALESCE(excluded.name, assessments.name),
                description=COALESCE(excluded.description, assessments.description),
                test_type=COALESCE(excluded.test_type, assessments.test_type),
                test_type_full=COALESCE(excluded.test_type_full, assessments.test_type_full),
                remote_support=COALESCE(excluded.remote_support, assessments.remote_support),
                adaptive_support=COALESCE(excluded.adaptive_support, assessments.adaptive_support),
                duration_minutes=COALESCE(excluded.duration_minutes, assessments.duration_minutes),
                job_levels=COALESCE(excluded.job_levels, assessments.job_levels),
                languages=COALESCE(excluded.languages, assessments.languages),
                downloads=COALESCE(excluded.downloads, assessments.downloads),
                source_catalog_page=COALESCE(excluded.source_catalog_page, assessments.source_catalog_page),
                last_updated_at=excluded.last_updated_at
            """,
            (
                data["assessment_id"],
                url,
                data.get("name"),
                data.get("description"),
                data.get("test_type"),
                data.get("test_type_full"),
                data.get("remote_support"),
                data.get("adaptive_support"),
                data.get("duration_minutes"),
                job_levels,
                languages,
                downloads,
                data.get("source_catalog_page"),
                data.get("discovered_at") or now_iso(),
                data.get("last_updated_at") or now_iso(),
            ),
        )
        self.conn.commit()

    def fetch_assessments(self) -> List[sqlite3.Row]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM assessments ORDER BY name")
        return cur.fetchall()

    def count_assessments(self) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM assessments")
        return cur.fetchone()[0]

    def close(self) -> None:
        self.conn.close()

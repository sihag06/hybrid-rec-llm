from __future__ import annotations

import logging
import urllib.robotparser
from dataclasses import dataclass

import structlog


logger = structlog.get_logger(__name__)


@dataclass
class RobotsManager:
    robots_url: str
    user_agent: str

    def __post_init__(self) -> None:
        self._parser = urllib.robotparser.RobotFileParser()

    def load(self) -> None:
        logger.info("robots.load.start", robots_url=self.robots_url)
        self._parser.set_url(self.robots_url)
        try:
            self._parser.read()
            logger.info("robots.load.success", can_fetch_all=self._parser.can_fetch(self.user_agent, "*"))
        except Exception as exc:  # pragma: no cover - network errors are logged
            logger.warning("robots.load.failed", error=str(exc))

    def is_allowed(self, url: str) -> bool:
        try:
            return self._parser.can_fetch(self.user_agent, url)
        except Exception as exc:  # pragma: no cover
            logger.warning("robots.check.error", url=url, error=str(exc))
            return False

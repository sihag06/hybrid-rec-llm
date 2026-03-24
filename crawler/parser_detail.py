from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

import structlog
from bs4 import BeautifulSoup, Tag

from crawler.storage import PARSE_PARSED, Storage
from crawler.utils import canonicalize_url, now_iso

logger = structlog.get_logger(__name__)

ALLOWED_TEST_TYPES = {"A", "B", "C", "D", "E", "K", "P", "S"}
STOP_LABELS = [
    "Job levels",
    "Job level",
    "Languages",
    "Language",
    "Assessment length",
    "Assessment Length",
    "Test Type",
    "Remote Testing",
    "Adaptive/IRT",
    "Adaptive",
    "Downloads",
]
STOP_LABELS_LOWER = [s.lower() for s in STOP_LABELS]
TEST_TYPE_LABELS = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _extract_text(soup: BeautifulSoup, selector: str) -> Optional[str]:
    node = soup.select_one(selector)
    if not node:
        return None
    text = _normalize(node.get_text(" ", strip=True))
    return text or None


def _find_label_node(soup: BeautifulSoup, label: str) -> Optional[Tag]:
    label_l = label.lower()
    candidates = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "div", "span", "strong", "dt", "th", "li"])
    for node in candidates:
        txt = _normalize(node.get_text(" ", strip=True)).lower()
        if txt == label_l or txt.startswith(label_l + ":") or txt.startswith(label_l):
            return node
    for node in candidates:
        txt = _normalize(node.get_text(" ", strip=True)).lower()
        if re.search(rf"\b{re.escape(label_l)}\b", txt):
            return node
    return None


def _extract_section_until(soup: BeautifulSoup, start_label: str, stop_labels: Iterable[str]) -> Optional[str]:
    start = _find_label_node(soup, start_label)
    if not start:
        return None

    chunks: List[str] = []

    start_txt = _normalize(start.get_text(" ", strip=True))
    if re.match(rf"^{re.escape(start_label)}\s*:", start_txt, flags=re.I):
        after = re.split(rf"^{re.escape(start_label)}\s*:\s*", start_txt, flags=re.I)[-1]
        if after:
            chunks.append(after)

    for node in start.find_all_next():
        if node == start:
            continue
        if not isinstance(node, Tag):
            continue

        node_txt = _normalize(node.get_text(" ", strip=True))
        if not node_txt:
            continue

        for stop in stop_labels:
            if re.match(rf"^{re.escape(stop)}\b", node_txt, flags=re.I):
                return _normalize(" ".join(chunks)) or None

        if node.name in {"p", "li"}:
            chunks.append(node_txt)
        elif node.name in {"div", "span"} and len(node_txt) > 40:
            chunks.append(node_txt)

    return _normalize(" ".join(chunks)) or None


def _extract_segment(text: str, label: str, stop_labels: Iterable[str]) -> Optional[str]:
    """Extract substring after a label up to the next stop label in raw text."""
    text_norm = _normalize(text)
    lower = text_norm.lower()
    label_l = label.lower()
    start = lower.find(label_l)
    if start == -1:
        return None
    start = start + len(label_l)
    while start < len(text_norm) and text_norm[start] in " :":
        start += 1
    stop_pos = len(text_norm)
    for stop in stop_labels:
        pos = lower.find(stop, start)
        if pos != -1 and pos < stop_pos:
            stop_pos = pos
    segment = text_norm[start:stop_pos].strip(" :-")
    return segment or None


def _extract_kv_value(soup: BeautifulSoup, label: str) -> Optional[str]:
    node = _find_label_node(soup, label)
    if not node:
        return None

    txt = _normalize(node.get_text(" ", strip=True))
    m = re.match(rf"^{re.escape(label)}\s*:\s*(.+)$", txt, flags=re.I)
    if m:
        return m.group(1).strip() or None

    remainder = re.sub(rf"^{re.escape(label)}\s*", "", txt, flags=re.I).strip(" :-")
    if remainder and remainder.lower() != label.lower():
        return remainder

    for sib in node.next_siblings:
        if isinstance(sib, Tag):
            v = _normalize(sib.get_text(" ", strip=True))
            if v:
                return v

    parent = node.parent if isinstance(node.parent, Tag) else None
    if parent:
        parent_txt = _normalize(parent.get_text(" ", strip=True))
        parent_remainder = re.sub(rf"\b{re.escape(label)}\b", "", parent_txt, flags=re.I).strip(" :-")
        if parent_remainder:
            return parent_remainder
        for sib in parent.find_next_siblings():
            v = _normalize(sib.get_text(" ", strip=True))
            if v:
                return v

    return None


def _extract_duration_minutes(soup: BeautifulSoup) -> Optional[int]:
    text = _normalize(soup.get_text(" ", strip=True))
    patterns = [
        r"minutes?\s*=\s*(\d+)",
        r"(\d+)\s*(?:minute|min)\b",
        r"completion time.*?(\d+)\s*(?:minute|min)\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                continue
    return None


def _extract_test_type_from_meta(soup: BeautifulSoup) -> Optional[str]:
    label = _find_label_node(soup, "Test Type")
    scope = label.parent if label and isinstance(label.parent, Tag) else label or soup

    tokens: List[str] = []
    for el in scope.find_all(["span", "button", "a"], limit=30):
        t = _normalize(el.get_text("", strip=True))
        if len(t) == 1 and t in ALLOWED_TEST_TYPES:
            tokens.append(t)
    if not tokens:
        for el in label.find_all_next(["span", "button", "a"], limit=30) if label else []:
            t = _normalize(el.get_text("", strip=True))
            if len(t) == 1 and t in ALLOWED_TEST_TYPES:
                tokens.append(t)
    if not tokens:
        return None
    out = []
    seen = set()
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return ",".join(out)


def _map_test_types_full(test_type: Optional[str]) -> Optional[str]:
    if not test_type:
        return None
    parts = []
    for token in test_type.split(","):
        token = token.strip()
        if not token:
            continue
        full = TEST_TYPE_LABELS.get(token)
        if full:
            parts.append(full)
    return ", ".join(parts) if parts else None


def _split_list(value: Optional[str]) -> Optional[list[str]]:
    if not value:
        return None
    parts = [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]
    return parts or None


def _is_positive_indicator(node: Tag) -> bool:
    if not node:
        return False
    attrs = " ".join(
        [
            " ".join(node.get("class", [])) if isinstance(node.get("class"), list) else str(node.get("class") or ""),
            str(node.get("aria-label") or ""),
            str(node.get("title") or ""),
            str(node.get("style") or ""),
        ]
    ).lower()
    positive_tokens = ["green", "yes", "true", "available", "supported", "active", "enabled", "tick", "check", "on"]
    return any(tok in attrs for tok in positive_tokens)


def _extract_boolean_from_meta(soup: BeautifulSoup, label_text: str) -> Optional[bool]:
    label = _find_label_node(soup, label_text)
    if not label:
        return None

    container = label.parent if isinstance(label.parent, Tag) else label
    for el in container.find_all(["span", "i", "svg", "img"], limit=20):
        if _is_positive_indicator(el):
            return True

    for el in label.find_all_next(["span", "i", "svg", "img"], limit=20):
        if _is_positive_indicator(el):
            return True

    return False


def extract_detail_fields(html: str) -> Dict:
    soup = BeautifulSoup(html, "lxml")

    title = _extract_text(soup, "h1") or _extract_text(soup, "title")
    full_text = _normalize(soup.get_text(" ", strip=True))
    description = _extract_segment(full_text, "description", STOP_LABELS_LOWER)
    if not description:
        description = _extract_section_until(soup, "Description", STOP_LABELS)

    job_levels_raw = _extract_kv_value(soup, "Job levels") or _extract_segment(full_text, "job levels", STOP_LABELS_LOWER)
    job_levels = _split_list(job_levels_raw)
    languages_raw = _extract_kv_value(soup, "Languages") or _extract_segment(full_text, "languages", STOP_LABELS_LOWER)
    languages = _split_list(languages_raw)

    duration = _extract_duration_minutes(soup)
    if duration is None:
        segment = _extract_segment(full_text, "assessment length", STOP_LABELS_LOWER)
        if segment:
            match = re.search(r"(\d+)\s*(?:minute|min)", segment, flags=re.I)
            if match:
                try:
                    duration = int(match.group(1))
                except Exception:
                    duration = None

    test_type = _extract_test_type_from_meta(soup)
    test_type_full = _map_test_types_full(test_type)

    remote_support = _extract_boolean_from_meta(soup, "Remote Testing")
    adaptive_support = _extract_boolean_from_meta(soup, "Adaptive/IRT")
    if adaptive_support is None:
        adaptive_support = _extract_boolean_from_meta(soup, "Adaptive")
    if adaptive_support is None:
        adaptive_support = _extract_boolean_from_meta(soup, "Adaptive/IRT Testing")

    downloads = []
    downloads_label = _find_label_node(soup, "Downloads")
    scope = downloads_label.parent if downloads_label and isinstance(downloads_label.parent, Tag) else soup
    for link in scope.find_all("a", href=True):
        text = _normalize(link.get_text(" ", strip=True))
        href = link["href"]
        if text and any(keyword in text.lower() for keyword in ["report", "fact sheet", "sample", "pdf", "download", "brochure"]):
            downloads.append({"text": text, "url": href})

    return {
        "name": title,
        "description": description,
        "test_type": test_type,
        "test_type_full": test_type_full,
        "remote_support": remote_support,
        "adaptive_support": adaptive_support,
        "duration_minutes": duration,
        "job_levels": job_levels,
        "languages": languages,
        "downloads": downloads or None,
    }


def parse_detail_page(html: str, url: str, storage: Storage) -> Dict:
    fields = extract_detail_fields(html)
    storage.upsert_assessment(
        {
            "url": canonicalize_url(url),
            **fields,
            "last_updated_at": now_iso(),
        }
    )
    storage.update_parse_status(url, PARSE_PARSED)
    logger.info("detail.parse.success", url=url, name=fields.get("name"))
    return fields

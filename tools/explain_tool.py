from __future__ import annotations

from typing import List, Optional

from schemas.query_plan import QueryPlan
from schemas.candidates import RankedList


def explain(plan: QueryPlan, ranked: RankedList, catalog_lookup) -> str:
    """
    Generate a simple textual explanation using catalog metadata.
    catalog_lookup: callable(assessment_id) -> dict with fields like name/url/test_type/duration.
    """
    lines: List[str] = []
    lines.append("Top recommendations:")
    for item in ranked.items:
        meta = catalog_lookup(item.assessment_id) or {}
        name = meta.get("name", item.assessment_id)
        url = meta.get("url", "")
        tt = meta.get("test_type_full") or meta.get("test_type")
        dur = meta.get("duration_minutes") or meta.get("duration")
        parts = [f"- {name}"]
        if tt:
            parts.append(f"[{tt}]")
        if dur:
            parts.append(f"~{dur} min")
        if url:
            parts.append(f"({url})")
        lines.append(" ".join(parts))
    # Brief note on matching
    lines.append("")
    lines.append("Matched intent: " + plan.intent)
    if plan.must_have_skills:
        lines.append("Key skills: " + ", ".join(plan.must_have_skills[:5]))
    if plan.duration and plan.duration.minutes:
        lines.append(f"Duration preference: {plan.duration.mode or 'TARGET'} {plan.duration.minutes} minutes")
    return "\n".join(lines)

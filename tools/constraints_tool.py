from __future__ import annotations

from typing import List, Optional, Dict
import math

from schemas.candidates import RankedList, RankedItem
from schemas.query_plan import QueryPlan


# Tunable weights / tolerances (small so relevance stays primary)
W_DURATION = 0.12
W_LANGUAGE = 0.08
W_FLAGS = 0.08
W_INTENT = 0.12
TOLERANCE_MINUTES = 20  # how far from target is still "ok"

INTENT_BEHAV_TYPES = {"Personality & Behavior", "Biodata & Situational Judgement", "Assessment Exercises", "Simulations"}
INTENT_TECH_TYPES = {"Knowledge & Skills", "Ability & Aptitude"}


def _safe_float(val) -> Optional[float]:
    try:
        if val is None:
            return None
        f = float(val)
        if math.isfinite(f):
            return f
    except Exception:
        return None
    return None


def _duration_score(duration: Optional[float], constraint) -> float:
    if constraint is None:
        return 0.0
    if duration is None:
        # Missing is neutral
        return 0.0
    target = constraint.minutes or 0
    if target <= 0:
        return 0.0
    diff = abs(duration - target)
    score = max(0.0, 1.0 - diff / TOLERANCE_MINUTES)
    if constraint.mode == "MAX" and duration > target:
        # small penalty if over max
        score = -score
    return score


def _language_score(plan: QueryPlan, meta_langs: List[str]) -> (float, bool):
    if not plan.language:
        return 0.0, True
    if not meta_langs:
        # missing metadata -> neutral, allow
        return 0.0, True
    match = any(plan.language.lower() in lang.lower() for lang in meta_langs)
    if match:
        return 1.0, True
    # treat language as soft penalty rather than hard drop
    return -1.0, True


def _flags_score(plan: QueryPlan, remote: Optional[bool], adaptive: Optional[bool]) -> (float, bool):
    score = 0.0
    flags = plan.flags or {}
    want_remote = flags.get("remote")
    want_adaptive = flags.get("adaptive")

    # Remote handling
    if want_remote is True:
        if remote is False:
            return -1.0, True  # soft penalty
        if remote is True:
            score += 1.0

    # Adaptive handling
    if want_adaptive is True:
        if adaptive is False:
            return -1.0, True  # soft penalty
        if adaptive is True:
            score += 1.0

    return score, True


def _intent_score(plan: QueryPlan, test_types: List[str]) -> float:
    if not test_types:
        return 0.0
    types_set = set(test_types)
    if plan.intent == "BEHAVIORAL":
        return 1.0 if types_set & INTENT_BEHAV_TYPES else 0.0
    if plan.intent == "TECH":
        return 1.0 if types_set & INTENT_TECH_TYPES else 0.0
    if plan.intent == "MIXED":
        beh = 1.0 if types_set & INTENT_BEHAV_TYPES else 0.0
        tech = 1.0 if types_set & INTENT_TECH_TYPES else 0.0
        return 0.5 * (beh + tech)
    return 0.0


def apply_constraints(plan: QueryPlan, ranked: RankedList, catalog_by_id: Dict[str, dict], k: int = 10) -> RankedList:
    """
    Deterministic, missingness-aware constraint layer.
    - Duration: soft boost/penalty; missing is neutral.
    - Language: soft penalty if mismatch; missing metadata is neutral.
    - Intent alignment: small boost when test_type matches intent.
    """
    rescored: List[RankedItem] = []
    for idx, item in enumerate(ranked.items):
        if item.assessment_id not in catalog_by_id:
            continue
        meta = catalog_by_id[item.assessment_id]
        duration = _safe_float(meta.get("duration_minutes") or meta.get("duration"))
        test_types = meta.get("test_type_full") or meta.get("test_type") or []
        if isinstance(test_types, str):
            test_types = [t.strip() for t in test_types.replace("/", ",").split(",") if t.strip()]
        meta_langs = meta.get("languages") or []
        if isinstance(meta_langs, str):
            meta_langs = [meta_langs]
        remote = meta.get("remote_support")
        adaptive = meta.get("adaptive_support")

        dur_s = _duration_score(duration, plan.duration)
        lang_s, allow_lang = _language_score(plan, meta_langs)
        flag_s, allow_flags = _flags_score(plan, remote, adaptive)
        intent_s = _intent_score(plan, test_types)

        if not allow_lang or not allow_flags:
            continue

        # Fallback if upstream left score as None: use a simple rank-based proxy.
        base_score = item.score if item.score is not None else 1.0 / (idx + 1)
        final_score = base_score
        final_score += W_DURATION * dur_s
        final_score += W_LANGUAGE * lang_s
        final_score += W_FLAGS * flag_s
        final_score += W_INTENT * intent_s

        debug = {
            "base_score": base_score,
            "duration": duration,
            "duration_score": W_DURATION * dur_s,
            "language": plan.language,
            "language_score": W_LANGUAGE * lang_s,
            "flags_score": W_FLAGS * flag_s,
            "intent_score": W_INTENT * intent_s,
            "test_types": test_types,
            "languages_meta": meta_langs,
            "remote": remote,
            "adaptive": adaptive,
            "final_score": final_score,
        }

        rescored.append(RankedItem(assessment_id=item.assessment_id, score=final_score, debug=debug))

    rescored.sort(key=lambda x: x.score, reverse=True)
    return RankedList(items=rescored[:k])

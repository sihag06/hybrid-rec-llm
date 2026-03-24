from __future__ import annotations

from typing import Iterable, List, Sequence, Set


def recall_at_k(ground_truth: Set[str], preds: Sequence[str], k: int) -> float:
    if not ground_truth:
        return 0.0
    topk = preds[:k]
    hits = len(ground_truth.intersection(topk))
    return hits / len(ground_truth)


def mrr_at_k(ground_truth: Set[str], preds: Sequence[str], k: int) -> float:
    if not ground_truth:
        return 0.0
    for idx, pid in enumerate(preds[:k], start=1):
        if pid in ground_truth:
            return 1.0 / idx
    return 0.0


def mean_metric(queries: Iterable[Set[str]], preds_list: Iterable[Sequence[str]], fn, k: int) -> float:
    scores = []
    for g, p in zip(queries, preds_list):
        scores.append(fn(g, p, k))
    return sum(scores) / len(scores) if scores else 0.0

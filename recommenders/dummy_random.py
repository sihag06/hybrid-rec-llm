from __future__ import annotations

import random
from typing import List, Sequence

from recommenders.base import Recommender


class DummyRandomRecommender(Recommender):
    def __init__(self, assessment_ids: Sequence[str], seed: int = 42) -> None:
        self.ids = list(assessment_ids)
        self.random = random.Random(seed)

    def recommend(self, query: str, k: int = 10) -> List[str]:
        if not self.ids:
            return []
        self.random.shuffle(self.ids)
        return self.ids[:k]

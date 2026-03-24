from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class Recommender(ABC):
    @abstractmethod
    def recommend(self, query: str, k: int = 10) -> List[str]:
        """Return top-k assessment_ids."""
        raise NotImplementedError

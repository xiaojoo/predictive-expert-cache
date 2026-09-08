from __future__ import annotations

from dataclasses import dataclass

from .cache import PredictiveExpertCache
from .types import ExpertId, ExpertPrediction


@dataclass(slots=True)
class PrefetchRequest:
    expert_id: ExpertId
    score: float
    priority: float


class ExpertScheduler:
    """
    High-level predictive_scheduler.

    Currently this class only generates prefetch requests.
    Actual data movement will be implemented later.
    """

    def __init__(
        self,
        cache: PredictiveExpertCache,
    ):
        self.cache = cache

    def plan_prefetch(
        self,
        current_experts: list[ExpertId],
    ) -> list[PrefetchRequest]:

        predictions = self.cache.prefetch_candidates(
            current_experts
        )

        return [
            PrefetchRequest(
                expert_id=p.expert_id,
                score=p.score,
                priority=p.score,
            )
            for p in predictions
        ]
from __future__ import annotations

from .lru import ExpertLRU
from .predictor import ExpertPredictor
from .types import (
    CacheConfig,
    CacheEntry,
    ExpertId,
    ExpertPrediction,
)


class PredictiveExpertCache:
    """
    Algorithm-level predictive expert cache.

    This class does NOT store actual model weights.

    It only manages:

        - expert access statistics
        - prediction
        - cache residency
        - LRU eviction
    """

    def __init__(
        self,
        config: CacheConfig | None = None,
    ):
        self.config = config or CacheConfig()

        self.predictor = ExpertPredictor(
            recent_window=self.config.recent_window,
            frequency_weight=self.config.frequency_weight,
            recency_weight=self.config.recency_weight,
            transition_weight=self.config.transition_weight,
            recency_decay=self.config.recency_decay,
        )

        self.lru = ExpertLRU[ExpertId, CacheEntry](
            self.config.capacity
        )

        self._step = 0

    # ---------------------------------------------------------
    # Observe router
    # ---------------------------------------------------------

    def observe(
        self,
        experts: list[ExpertId],
    ) -> None:

        self.predictor.observe(experts)

        self._step = self.predictor.step

        for expert_id in experts:
            entry = self.lru.peek(expert_id)

            if entry is not None:
                entry.last_access_step = self._step
                entry.access_count += 1
                entry.hit_count += 1

                # Touch LRU ordering.
                self.lru.get(expert_id)

    # ---------------------------------------------------------
    # Cache expert
    # ---------------------------------------------------------

    def insert(
        self,
        expert_id: ExpertId,
        size_bytes: int = 0,
        location: str = "memory",
    ) -> ExpertId | None:

        existing = self.lru.peek(expert_id)

        if existing is not None:
            existing.last_access_step = self._step
            existing.access_count += 1

            self.lru.get(expert_id)

            return None

        entry = CacheEntry(
            expert_id=expert_id,
            inserted_step=self._step,
            last_access_step=self._step,
            access_count=1,
            size_bytes=size_bytes,
            location=location,
        )

        evicted = self.lru.put(
            expert_id,
            entry,
        )

        return evicted

    # ---------------------------------------------------------
    # Lookup
    # ---------------------------------------------------------

    def get(
        self,
        expert_id: ExpertId,
    ) -> CacheEntry | None:

        entry = self.lru.get(expert_id)

        if entry is None:
            stats = self.predictor.stats.get(
                expert_id
            )

            if stats is not None:
                stats.miss_count += 1

            return None

        entry.last_access_step = self._step
        entry.access_count += 1
        entry.hit_count += 1

        return entry

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    def predict(
        self,
        current_experts: list[ExpertId],
        top_k: int | None = None,
    ) -> list[ExpertPrediction]:

        if top_k is None:
            top_k = self.config.prediction_top_k

        predictions = self.predictor.predict(
            current_experts=current_experts,
            top_k=top_k,
        )

        return [
            prediction
            for prediction in predictions
            if prediction.score
            >= self.config.min_prediction_score
        ]

    # ---------------------------------------------------------
    # Prefetch candidates
    # ---------------------------------------------------------

    def prefetch_candidates(
        self,
        current_experts: list[ExpertId],
    ) -> list[ExpertPrediction]:

        predictions = self.predict(
            current_experts
        )

        return [
            prediction
            for prediction in predictions
            if prediction.expert_id
            not in self.lru
        ]

    # ---------------------------------------------------------
    # Cache state
    # ---------------------------------------------------------

    def contains(
        self,
        expert_id: ExpertId,
    ) -> bool:

        return expert_id in self.lru

    @property
    def size(self) -> int:
        return len(self.lru)

    @property
    def capacity(self) -> int:
        return self.lru.capacity

    def cached_experts(self) -> list[ExpertId]:
        return self.lru.keys()

    def clear(self) -> None:
        self.lru.clear()
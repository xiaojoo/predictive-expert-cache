from __future__ import annotations

from collections.abc import Iterable

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

    This layer stores expert metadata only.
    Actual model weights remain outside the cache core.

    Responsibilities:

        - routing statistics
        - prediction
        - cache residency
        - LRU eviction
        - cache hit/miss statistics
    """

    def __init__(
        self,
        config: CacheConfig | None = None,
    ) -> None:

        self.config = (
            config
            or CacheConfig()
        )

        self.predictor = ExpertPredictor(
            recent_window=(
                self.config.recent_window
            ),
            frequency_weight=(
                self.config.frequency_weight
            ),
            recency_weight=(
                self.config.recency_weight
            ),
            transition_weight=(
                self.config.transition_weight
            ),
            recency_decay=(
                self.config.recency_decay
            ),
        )

        self.lru = ExpertLRU[
            ExpertId,
            CacheEntry,
        ](
            self.config.capacity
        )

        self._step = 0

    # ---------------------------------------------------------
    # Observe router
    # ---------------------------------------------------------

    def observe(
        self,
        experts: Iterable[ExpertId],
    ) -> None:
        """
        Record a router event.

        Important:
        observing a routed expert does NOT mean the expert
        was present in this cache.
        """

        self.predictor.observe(
            experts
        )

        self._step = (
            self.predictor.step
        )

    # ---------------------------------------------------------
    # Insert
    # ---------------------------------------------------------

    def insert(
        self,
        expert_id: ExpertId,
        size_bytes: int = 0,
        location: str = "memory",
    ) -> ExpertId | None:

        if size_bytes < 0:
            raise ValueError(
                "size_bytes must be >= 0"
            )

        if not location:
            raise ValueError(
                "location must not be empty"
            )

        existing = self.lru.peek(
            expert_id
        )

        if existing is not None:
            existing.last_access_step = (
                self._step
            )

            self.lru.get(
                expert_id
            )

            return None

        entry = CacheEntry(
            expert_id=expert_id,
            inserted_step=self._step,
            last_access_step=self._step,
            size_bytes=size_bytes,
            location=location,
        )

        return self.lru.put(
            expert_id,
            entry,
        )

    # ---------------------------------------------------------
    # Lookup
    # ---------------------------------------------------------

    def get(
        self,
        expert_id: ExpertId,
    ) -> CacheEntry | None:

        entry = self.lru.get(
            expert_id
        )

        if entry is None:
            self.predictor.record_miss(
                expert_id
            )

            return None

        entry.last_access_step = (
            self._step
        )

        entry.access_count += 1
        entry.hit_count += 1

        self.predictor.record_hit(
            expert_id
        )

        return entry

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    def predict(
        self,
        current_experts: Iterable[ExpertId],
        top_k: int | None = None,
    ) -> list[ExpertPrediction]:

        if top_k is None:
            top_k = (
                self.config.prediction_top_k
            )

        if top_k <= 0:
            return []

        predictions = (
            self.predictor.predict(
                current_experts,
                top_k=top_k,
            )
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
        current_experts: Iterable[ExpertId],
    ) -> list[ExpertPrediction]:

        return [
            prediction
            for prediction in self.predict(
                current_experts
            )
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

    def cached_experts(
        self,
    ) -> list[ExpertId]:
        return self.lru.keys()

    def clear(self) -> None:
        self.lru.clear()
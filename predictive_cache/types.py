from __future__ import annotations

from dataclasses import dataclass


ExpertId = int


@dataclass(slots=True)
class ExpertStats:
    """Runtime statistics for a single expert."""

    expert_id: ExpertId
    frequency: int = 0
    last_seen_step: int = -1
    first_seen_step: int = -1
    hit_count: int = 0
    miss_count: int = 0

    @property
    def access_count(self) -> int:
        """Number of cache lookups recorded for this expert."""
        return self.hit_count + self.miss_count

    @property
    def hit_rate(self) -> float:
        """Cache hit rate for recorded lookups."""
        total = self.access_count

        if total == 0:
            return 0.0

        return self.hit_count / total


@dataclass(slots=True)
class ExpertPrediction:
    """Prediction result for a future expert."""

    expert_id: ExpertId
    score: float
    frequency_score: float = 0.0
    recency_score: float = 0.0
    transition_score: float = 0.0


@dataclass(slots=True)
class CacheEntry:
    """Metadata for one cached expert."""

    expert_id: ExpertId
    inserted_step: int
    last_access_step: int

    access_count: int = 0
    hit_count: int = 0
    miss_count: int = 0

    size_bytes: int = 0
    location: str = "memory"

    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count

        if total == 0:
            return 0.0

        return self.hit_count / total


@dataclass(slots=True)
class CacheConfig:
    """Configuration of PredictiveExpertCache."""

    capacity: int = 8
    prediction_top_k: int = 4
    recent_window: int = 32

    frequency_weight: float = 0.30
    recency_weight: float = 0.20
    transition_weight: float = 0.50

    recency_decay: float = 8.0
    min_prediction_score: float = 0.0

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be > 0")

        if self.prediction_top_k <= 0:
            raise ValueError("prediction_top_k must be > 0")

        if self.recent_window <= 0:
            raise ValueError("recent_window must be > 0")

        if self.recency_decay <= 0:
            raise ValueError("recency_decay must be > 0")

        weights = (
            self.frequency_weight,
            self.recency_weight,
            self.transition_weight,
        )

        if any(weight < 0 for weight in weights):
            raise ValueError(
                "prediction weights must be >= 0"
            )

        if sum(weights) <= 0:
            raise ValueError(
                "prediction weights must have positive sum"
            )
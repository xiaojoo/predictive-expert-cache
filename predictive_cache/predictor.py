from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from collections.abc import Iterable

from .types import (
    ExpertId,
    ExpertPrediction,
    ExpertStats,
)


class ExpertPredictor:
    """
    Predict future experts from:

    1. Frequency
    2. Recency
    3. Transition probability
    4. Recent access window
    """

    def __init__(
        self,
        recent_window: int = 32,
        frequency_weight: float = 0.30,
        recency_weight: float = 0.20,
        transition_weight: float = 0.50,
        recency_decay: float = 8.0,
    ) -> None:
        if recent_window <= 0:
            raise ValueError(
                "recent_window must be > 0"
            )

        if recency_decay <= 0:
            raise ValueError(
                "recency_decay must be > 0"
            )

        weights = (
            frequency_weight,
            recency_weight,
            transition_weight,
        )

        if any(weight < 0 for weight in weights):
            raise ValueError(
                "prediction weights must be >= 0"
            )

        weight_sum = sum(weights)

        if weight_sum <= 0:
            raise ValueError(
                "prediction weights must have positive sum"
            )

        self.recent_window = recent_window

        # Normalize weights so callers can provide
        # arbitrary non-negative weights.
        self.frequency_weight = (
            frequency_weight / weight_sum
        )

        self.recency_weight = (
            recency_weight / weight_sum
        )

        self.transition_weight = (
            transition_weight / weight_sum
        )

        self.recency_decay = recency_decay

        self._step = 0
        self._max_frequency = 0

        self._stats: dict[
            ExpertId,
            ExpertStats,
        ] = {}

        self._transitions: dict[
            ExpertId,
            Counter[ExpertId],
        ] = defaultdict(Counter)

        self._transition_totals: dict[ExpertId, int] = defaultdict(int)

        self._recent: deque[ExpertId] = deque(
            maxlen=recent_window
        )

        self._previous_experts: list[
            ExpertId
        ] = []

    # ---------------------------------------------------------
    # Observe
    # ---------------------------------------------------------

    def observe(
        self,
        experts: Iterable[ExpertId],
    ) -> None:
        """
        Observe one router event.

        Duplicate expert IDs inside the same routing event
        are collapsed while preserving order.
        """

        current = list(
            dict.fromkeys(experts)
        )

        if not current:
            return

        current_step = self._step

        for expert_id in current:
            stats = self._stats.get(
                expert_id
            )

            stats = self._stats.get(
                expert_id
            )

            if stats is None:
                stats = ExpertStats(
                    expert_id=expert_id,
                    first_seen_step=current_step,
                )

                self._stats[expert_id] = stats
            elif stats.last_seen_step >= 0:
                distance = (
                        current_step
                        - stats.last_seen_step
                )

                stats.distance_sum += distance
                stats.distance_count += 1

            stats.frequency += 1
            stats.last_seen_step = current_step

            if stats.frequency > self._max_frequency:
                self._max_frequency = stats.frequency

            self._recent.append(
                expert_id
            )

        # -----------------------------------------------------
        # Build transitions
        # -----------------------------------------------------

        if self._previous_experts:
            for previous_expert in self._previous_experts:
                transitions = self._transitions[previous_expert]

                for current_expert in current:
                    transitions[current_expert] += 1
                    self._transition_totals[previous_expert] += 1

        self._previous_experts = current

        self._step += 1

    # ---------------------------------------------------------
    # Cache statistics
    # ---------------------------------------------------------

    def record_hit(
        self,
        expert_id: ExpertId,
    ) -> None:
        """
        Record a successful cache lookup.

        This does not affect routing frequency.
        """

        stats = self._stats.get(
            expert_id
        )

        if stats is None:
            stats = ExpertStats(
                expert_id=expert_id
            )

            self._stats[expert_id] = stats

        stats.hit_count += 1

    def record_miss(
        self,
        expert_id: ExpertId,
    ) -> None:
        """
        Record a failed cache lookup.

        This does not affect routing frequency.
        """

        stats = self._stats.get(
            expert_id
        )

        if stats is None:
            stats = ExpertStats(
                expert_id=expert_id
            )

            self._stats[expert_id] = stats

        stats.miss_count += 1

    # ---------------------------------------------------------
    # Transition probability
    # ---------------------------------------------------------

    def transition_probability(
            self,
            current_expert: ExpertId,
            next_expert: ExpertId,
    ) -> float:
        transitions = self._transitions.get(
            current_expert
        )

        if transitions is None:
            return 0.0

        total = self._transition_totals.get(
            current_expert,
            0,
        )

        if total == 0:
            return 0.0

        count = transitions.get(
            next_expert,
            0,
        )

        return count / total

    # ---------------------------------------------------------
    # Frequency score
    # ---------------------------------------------------------

    def _frequency_score(
            self,
            expert_id: ExpertId,
    ) -> float:

        if self._max_frequency == 0:
            return 0.0

        stats = self._stats.get(
            expert_id
        )

        if stats is None:
            return 0.0

        return (
                stats.frequency
                / self._max_frequency
        )

    # ---------------------------------------------------------
    # Recency score
    # ---------------------------------------------------------

    def _recency_score(
        self,
        expert_id: ExpertId,
    ) -> float:

        stats = self._stats.get(
            expert_id
        )

        if (
            stats is None
            or stats.last_seen_step < 0
        ):
            return 0.0

        age = (
            self._step
            - stats.last_seen_step
        )

        return math.exp(
            -age / self.recency_decay
        )

    # ---------------------------------------------------------
    # Transition score
    # ---------------------------------------------------------

    def _transition_score(
        self,
        current_experts: list[ExpertId],
        candidate: ExpertId,
    ) -> float:

        if not current_experts:
            return 0.0

        return max(
            (
                self.transition_probability(
                    current,
                    candidate,
                )
                for current in current_experts
            ),
            default=0.0,
        )

    # ---------------------------------------------------------
    # Predict
    # ---------------------------------------------------------

    def predict(
        self,
        current_experts: Iterable[ExpertId],
        top_k: int = 4,
    ) -> list[ExpertPrediction]:

        if top_k <= 0:
            return []

        current = list(
            dict.fromkeys(
                current_experts
            )
        )

        candidate_ids = set(
            self._stats
        )

        # Current experts don't need
        # to be prefetched.
        candidate_ids.difference_update(
            current
        )

        predictions: list[
            ExpertPrediction
        ] = []

        for expert_id in candidate_ids:
            frequency_score = (
                self._frequency_score(
                    expert_id
                )
            )

            recency_score = (
                self._recency_score(
                    expert_id
                )
            )

            transition_score = (
                self._transition_score(
                    current,
                    expert_id,
                )
            )

            stats = self._stats[expert_id]

            distance = stats.average_distance
            estimated_distance = distance

            score = (
                self.frequency_weight
                * frequency_score
                + self.recency_weight
                * recency_score
                + self.transition_weight
                * transition_score
            )

            predictions.append(
                ExpertPrediction(
                    expert_id=expert_id,
                    score=score,
                    frequency_score=frequency_score,
                    recency_score=recency_score,
                    transition_score=transition_score,
                    distance=distance,
                    estimated_distance=estimated_distance,
                )
            )

        # Deterministic ordering is important
        # for stable prefetch decisions.
        predictions.sort(
            key=lambda prediction: (
                -prediction.score,
                prediction.expert_id,
            )
        )

        return predictions[:top_k]

    # ---------------------------------------------------------
    # Properties
    # ---------------------------------------------------------

    @property
    def step(self) -> int:
        return self._step

    @property
    def stats(
        self,
    ) -> dict[ExpertId, ExpertStats]:
        return self._stats

    @property
    def transitions(
        self,
    ) -> dict[
        ExpertId,
        Counter[ExpertId],
    ]:
        return self._transitions

    @property
    def recent(
        self,
    ) -> list[ExpertId]:
        return list(self._recent)
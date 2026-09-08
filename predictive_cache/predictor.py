from __future__ import annotations

import math
from collections import Counter, defaultdict, deque

from .types import (
    ExpertId,
    ExpertPrediction,
    ExpertStats,
)


class ExpertPredictor:
    """
    Predict future experts based on:

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
    ):
        if recent_window <= 0:
            raise ValueError("recent_window must be > 0")

        self.recent_window = recent_window

        self.frequency_weight = frequency_weight
        self.recency_weight = recency_weight
        self.transition_weight = transition_weight

        self.recency_decay = recency_decay

        self._step = 0

        self._stats: dict[ExpertId, ExpertStats] = {}

        self._transitions: dict[
            ExpertId,
            Counter[ExpertId],
        ] = defaultdict(Counter)

        self._recent: deque[ExpertId] = deque(
            maxlen=recent_window
        )

        self._previous_experts: list[ExpertId] = []

    # ---------------------------------------------------------
    # Observe
    # ---------------------------------------------------------

    def observe(self, experts: list[ExpertId]) -> None:
        """
        Observe experts selected during one routing event.

        Example:

            observe([12, 41, 7, 99])
        """

        if not experts:
            return

        current_step = self._step

        # Update expert statistics.
        for expert_id in experts:
            stats = self._stats.get(expert_id)

            if stats is None:
                stats = ExpertStats(
                    expert_id=expert_id,
                    first_seen_step=current_step,
                )

                self._stats[expert_id] = stats

            stats.frequency += 1
            stats.last_seen_step = current_step

            self._recent.append(expert_id)

        # -----------------------------------------------------
        # Build transitions
        # -----------------------------------------------------

        if self._previous_experts:
            previous = self._previous_experts
            current = experts

            for prev_expert in previous:
                for current_expert in current:
                    if prev_expert != current_expert:
                        self._transitions[prev_expert][current_expert] += 1

        self._previous_experts = list(experts)

        self._step += 1

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

        if not transitions:
            return 0.0

        total = sum(transitions.values())

        if total == 0:
            return 0.0

        return transitions[next_expert] / total

    # ---------------------------------------------------------
    # Frequency
    # ---------------------------------------------------------

    def _frequency_score(
        self,
        expert_id: ExpertId,
    ) -> float:

        if not self._stats:
            return 0.0

        max_frequency = max(
            stats.frequency
            for stats in self._stats.values()
        )

        if max_frequency == 0:
            return 0.0

        stats = self._stats.get(expert_id)

        if stats is None:
            return 0.0

        return stats.frequency / max_frequency

    # ---------------------------------------------------------
    # Recency
    # ---------------------------------------------------------

    def _recency_score(
        self,
        expert_id: ExpertId,
    ) -> float:

        stats = self._stats.get(expert_id)

        if stats is None:
            return 0.0

        if stats.last_seen_step < 0:
            return 0.0

        age = self._step - stats.last_seen_step

        return math.exp(
            -age / self.recency_decay
        )

    # ---------------------------------------------------------
    # Transition
    # ---------------------------------------------------------

    def _transition_score(
        self,
        current_experts: list[ExpertId],
        candidate: ExpertId,
    ) -> float:

        if not current_experts:
            return 0.0

        probabilities = [
            self.transition_probability(
                current,
                candidate,
            )
            for current in current_experts
        ]

        return max(probabilities, default=0.0)

    # ---------------------------------------------------------
    # Predict
    # ---------------------------------------------------------

    def predict(
        self,
        current_experts: list[ExpertId],
        top_k: int = 4,
    ) -> list[ExpertPrediction]:

        if top_k <= 0:
            return []

        candidate_ids = set(self._stats.keys())

        # Current experts don't need to be prefetched.
        candidate_ids.difference_update(
            current_experts
        )

        predictions: list[ExpertPrediction] = []

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
                    current_experts,
                    expert_id,
                )
            )

            score = (
                self.frequency_weight
                * frequency_score
                +
                self.recency_weight
                * recency_score
                +
                self.transition_weight
                * transition_score
            )

            predictions.append(
                ExpertPrediction(
                    expert_id=expert_id,
                    score=score,
                    frequency_score=frequency_score,
                    recency_score=recency_score,
                    transition_score=transition_score,
                )
            )

        predictions.sort(
            key=lambda x: x.score,
            reverse=True,
        )

        return predictions[:top_k]

    # ---------------------------------------------------------
    # Properties
    # ---------------------------------------------------------

    @property
    def step(self) -> int:
        return self._step

    @property
    def stats(self) -> dict[ExpertId, ExpertStats]:
        return self._stats

    @property
    def transitions(
        self,
    ) -> dict[ExpertId, Counter[ExpertId]]:

        return self._transitions

    @property
    def recent(self) -> list[ExpertId]:
        return list(self._recent)
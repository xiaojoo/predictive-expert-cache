from __future__ import annotations

from dataclasses import dataclass

from .access_pattern import (
    ExpertAccessPattern,
    build_access_patterns,
)
from .history import RoutingHistory
from .transition_probability import (
    ExpertTransitionProbability,
)


@dataclass(frozen=True, slots=True)
class ExpertPredictionFeatures:
    """
    Historical prediction features for one expert.

    These features describe the historical behavior of an expert.
    Transition probability is candidate-specific and is therefore
    represented separately by ``ExpertCandidateFeatures``.
    """

    expert_id: int
    frequency: int
    first_seen_step: int
    last_seen_step: int
    recency: int
    layers: frozenset[int]

    def __post_init__(self) -> None:
        if self.expert_id < 0:
            raise ValueError("expert_id must be >= 0")

        if self.frequency <= 0:
            raise ValueError("frequency must be > 0")

        if self.first_seen_step < 0:
            raise ValueError(
                "first_seen_step must be >= 0"
            )

        if self.last_seen_step < self.first_seen_step:
            raise ValueError(
                "last_seen_step must be >= first_seen_step"
            )

        if self.recency < 0:
            raise ValueError("recency must be >= 0")

        if not self.layers:
            raise ValueError("layers must not be empty")

        if any(layer_id < 0 for layer_id in self.layers):
            raise ValueError("layer_id must be >= 0")


@dataclass(frozen=True, slots=True)
class ExpertCandidateFeatures:
    """
    Prediction features for an expert candidate.

    This combines historical features with the transition
    probability from the currently active experts.
    """

    expert_id: int
    frequency: int
    first_seen_step: int
    last_seen_step: int
    recency: int
    layers: frozenset[int]
    transition_probability: float

    def __post_init__(self) -> None:
        if self.expert_id < 0:
            raise ValueError("expert_id must be >= 0")

        if self.frequency <= 0:
            raise ValueError("frequency must be > 0")

        if self.first_seen_step < 0:
            raise ValueError(
                "first_seen_step must be >= 0"
            )

        if self.last_seen_step < self.first_seen_step:
            raise ValueError(
                "last_seen_step must be >= first_seen_step"
            )

        if self.recency < 0:
            raise ValueError("recency must be >= 0")

        if not self.layers:
            raise ValueError("layers must not be empty")

        if any(layer_id < 0 for layer_id in self.layers):
            raise ValueError("layer_id must be >= 0")

        if not 0.0 <= self.transition_probability <= 1.0:
            raise ValueError(
                "transition_probability must be between 0 and 1"
            )


def _pattern_to_features(
    pattern: ExpertAccessPattern,
    current_step: int,
) -> ExpertPredictionFeatures:
    """
    Convert one historical access pattern into prediction features.
    """

    if current_step < 0:
        raise ValueError("current_step must be >= 0")

    if current_step < pattern.last_seen_step:
        raise ValueError(
            "current_step must be >= last_seen_step"
        )

    return ExpertPredictionFeatures(
        expert_id=pattern.expert_id,
        frequency=pattern.access_count,
        first_seen_step=pattern.first_seen_step,
        last_seen_step=pattern.last_seen_step,
        recency=current_step - pattern.last_seen_step,
        layers=pattern.layers,
    )


def build_expert_features(
    history: RoutingHistory,
    current_step: int,
) -> dict[int, ExpertPredictionFeatures]:
    """
    Build historical prediction features for every known expert.

    ``recency`` is defined as:

        current_step - last_seen_step
    """

    patterns = build_access_patterns(history)

    return {
        expert_id: _pattern_to_features(
            pattern,
            current_step,
        )
        for expert_id, pattern in patterns.items()
    }


def get_transition_probability(
    transition_probabilities: dict[
        tuple[int, int],
        ExpertTransitionProbability,
    ],
    current_experts: tuple[int, ...] | list[int],
    candidate_expert_id: int,
) -> float:
    """
    Get the transition probability for one candidate expert.

    When multiple current experts exist, the strongest transition
    probability is used:

        max(P(candidate | current_expert))

    If no transition exists, return 0.0.
    """

    if candidate_expert_id < 0:
        raise ValueError(
            "candidate_expert_id must be >= 0"
        )

    if any(expert_id < 0 for expert_id in current_experts):
        raise ValueError(
            "current expert id must be >= 0"
        )

    probabilities = [
        transition.probability
        for (source_expert_id, target_expert_id), transition
        in transition_probabilities.items()
        if target_expert_id == candidate_expert_id
        and source_expert_id in current_experts
    ]

    if not probabilities:
        return 0.0

    return max(probabilities)


def build_candidate_features(
    history: RoutingHistory,
    current_step: int,
    current_experts: tuple[int, ...] | list[int],
    transition_probabilities: dict[
        tuple[int, int],
        ExpertTransitionProbability,
    ],
) -> dict[int, ExpertCandidateFeatures]:
    """
    Build prediction features for all known expert candidates.

    Historical features come from the routing history.

    Transition probability is computed relative to the current
    expert set and uses the strongest observed transition:

        max(P(candidate | current_expert))
    """

    historical_features = build_expert_features(
        history,
        current_step,
    )

    return {
        expert_id: ExpertCandidateFeatures(
            expert_id=features.expert_id,
            frequency=features.frequency,
            first_seen_step=features.first_seen_step,
            last_seen_step=features.last_seen_step,
            recency=features.recency,
            layers=features.layers,
            transition_probability=get_transition_probability(
                transition_probabilities,
                current_experts,
                expert_id,
            ),
        )
        for expert_id, features in historical_features.items()
    }
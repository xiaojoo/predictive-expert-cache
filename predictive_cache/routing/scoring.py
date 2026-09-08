from __future__ import annotations

import math
from dataclasses import dataclass

from .normalization import NormalizedExpertFeatures


@dataclass(frozen=True, slots=True)
class PredictionScore:
    """
    Prediction score for one expert.

    The score is composed of three explainable components:

        frequency_score
        recency_score
        transition_score
    """

    expert_id: int
    score: float
    frequency_score: float
    recency_score: float
    transition_score: float

    def __post_init__(self) -> None:
        if self.expert_id < 0:
            raise ValueError(
                "expert_id must be >= 0"
            )

        if not 0.0 <= self.frequency_score <= 1.0:
            raise ValueError(
                "frequency_score must be between 0 and 1"
            )

        if not 0.0 <= self.recency_score <= 1.0:
            raise ValueError(
                "recency_score must be between 0 and 1"
            )

        if not 0.0 <= self.transition_score <= 1.0:
            raise ValueError(
                "transition_score must be between 0 and 1"
            )

        if not 0.0 <= self.score <= 1.0:
            raise ValueError(
                "score must be between 0 and 1"
            )


def recency_score(
    recency: int,
    recency_decay: float,
) -> float:
    """
    Convert recency distance into a normalized score.

    Formula:

        recency_score =
            exp(-recency / recency_decay)

    Therefore:

        recency = 0
            -> 1.0

        recency = recency_decay
            -> exp(-1)

        larger recency
            -> smaller score
    """

    if recency < 0:
        raise ValueError(
            "recency must be >= 0"
        )

    if recency_decay <= 0:
        raise ValueError(
            "recency_decay must be > 0"
        )

    return math.exp(
        -recency / recency_decay
    )


def calculate_prediction_score(
    features: NormalizedExpertFeatures,
    frequency_weight: float,
    recency_weight: float,
    transition_weight: float,
    recency_decay: float,
) -> PredictionScore:
    """
    Calculate the weighted prediction score for one expert.

    Formula:

        score =
            frequency_weight * frequency_score
          + recency_weight * recency_score
          + transition_weight * transition_probability

    The weights are normalized by their total so that the final
    score remains in [0, 1] even when callers provide weights
    that do not sum exactly to 1.
    """

    weights = (
        frequency_weight,
        recency_weight,
        transition_weight,
    )

    if any(weight < 0 for weight in weights):
        raise ValueError(
            "prediction weights must be >= 0"
        )

    total_weight = sum(weights)

    if total_weight <= 0:
        raise ValueError(
            "prediction weights must have positive sum"
        )

    current_recency_score = recency_score(
        recency=features.recency,
        recency_decay=recency_decay,
    )

    normalized_frequency_weight = (
        frequency_weight / total_weight
    )
    normalized_recency_weight = (
        recency_weight / total_weight
    )
    normalized_transition_weight = (
        transition_weight / total_weight
    )

    score = (
        normalized_frequency_weight
        * features.frequency_score
        + normalized_recency_weight
        * current_recency_score
        + normalized_transition_weight
        * features.transition_probability
    )

    return PredictionScore(
        expert_id=features.expert_id,
        score=score,
        frequency_score=features.frequency_score,
        recency_score=current_recency_score,
        transition_score=features.transition_probability,
    )
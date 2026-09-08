from __future__ import annotations

from dataclasses import dataclass

from .features import ExpertCandidateFeatures


@dataclass(frozen=True, slots=True)
class NormalizedExpertFeatures:
    """
    Normalized prediction features for one expert.

    Frequency is normalized to [0, 1].

    Recency is intentionally kept as the raw step distance.
    It will be transformed into a recency score during the
    prediction scoring stage.

    Transition probability is already naturally bounded
    in [0, 1], so it is preserved unchanged.
    """

    expert_id: int
    frequency_score: float
    recency: int
    transition_probability: float

    def __post_init__(self) -> None:
        if self.expert_id < 0:
            raise ValueError("expert_id must be >= 0")

        if not 0.0 <= self.frequency_score <= 1.0:
            raise ValueError(
                "frequency_score must be between 0 and 1"
            )

        if self.recency < 0:
            raise ValueError("recency must be >= 0")

        if not 0.0 <= self.transition_probability <= 1.0:
            raise ValueError(
                "transition_probability must be between 0 and 1"
            )


def normalize_features(
    features: dict[int, ExpertCandidateFeatures],
) -> dict[int, NormalizedExpertFeatures]:
    """
    Normalize prediction features for all known experts.

    Frequency normalization:

        frequency_score =
            frequency / max_frequency

    Therefore the most frequently accessed expert receives
    a frequency score of 1.0.

    Recency remains unchanged because its transformation into
    a score depends on the configured recency decay and belongs
    to the prediction scoring stage.

    Transition probability is already in [0, 1] and is preserved.
    """

    if not features:
        return {}

    max_frequency = max(
        feature.frequency
        for feature in features.values()
    )

    if max_frequency <= 0:
        raise ValueError(
            "max_frequency must be > 0"
        )

    return {
        expert_id: NormalizedExpertFeatures(
            expert_id=feature.expert_id,
            frequency_score=(
                feature.frequency / max_frequency
            ),
            recency=feature.recency,
            transition_probability=(
                feature.transition_probability
            ),
        )
        for expert_id, feature in features.items()
    }
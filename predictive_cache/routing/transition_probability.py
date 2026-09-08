from __future__ import annotations

from dataclasses import dataclass

from .transition import ExpertTransition


@dataclass(frozen=True, slots=True)
class ExpertTransitionProbability:
    """
    Conditional probability of one expert following another.
    """

    source_expert_id: int
    target_expert_id: int
    probability: float

    def __post_init__(self) -> None:
        if self.source_expert_id < 0:
            raise ValueError(
                "source_expert_id must be >= 0"
            )

        if self.target_expert_id < 0:
            raise ValueError(
                "target_expert_id must be >= 0"
            )

        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(
                "probability must be between 0 and 1"
            )


def build_transition_probabilities(
    transitions: dict[
        tuple[int, int],
        ExpertTransition,
    ],
) -> dict[
    tuple[int, int],
    ExpertTransitionProbability,
]:
    """
    Convert transition counts into conditional probabilities.

    For each source expert:

        P(target | source)
            = count(source -> target)
              / total transitions from source
    """

    totals: dict[int, int] = {}

    for transition in transitions.values():
        source = transition.source_expert_id

        totals[source] = (
            totals.get(source, 0)
            + transition.count
        )

    probabilities: dict[
        tuple[int, int],
        ExpertTransitionProbability,
    ] = {}

    for key, transition in transitions.items():
        source = transition.source_expert_id

        total = totals[source]

        probabilities[key] = ExpertTransitionProbability(
            source_expert_id=source,
            target_expert_id=transition.target_expert_id,
            probability=transition.count / total,
        )

    return probabilities
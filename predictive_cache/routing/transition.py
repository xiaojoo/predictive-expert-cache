from __future__ import annotations

from dataclasses import dataclass

from .history import RoutingHistory


@dataclass(frozen=True, slots=True)
class ExpertTransition:
    """
    Transition count between two experts.
    """

    source_expert_id: int
    target_expert_id: int
    count: int

    def __post_init__(self) -> None:
        if self.source_expert_id < 0:
            raise ValueError(
                "source_expert_id must be >= 0"
            )

        if self.target_expert_id < 0:
            raise ValueError(
                "target_expert_id must be >= 0"
            )

        if self.count <= 0:
            raise ValueError("count must be > 0")


def build_expert_transitions(
    history: RoutingHistory,
) -> dict[tuple[int, int], ExpertTransition]:
    """
    Build expert-to-expert transition counts.

    Transitions are extracted only from experts inside the
    same RoutingEvent and follow the order of expert_ids.
    """

    transitions: dict[
        tuple[int, int],
        ExpertTransition,
    ] = {}

    for event in history.events():
        expert_ids = event.expert_ids

        for source_expert_id, target_expert_id in zip(
            expert_ids,
            expert_ids[1:],
        ):
            key = (
                source_expert_id,
                target_expert_id,
            )

            existing = transitions.get(key)

            if existing is None:
                transitions[key] = ExpertTransition(
                    source_expert_id=source_expert_id,
                    target_expert_id=target_expert_id,
                    count=1,
                )
                continue

            transitions[key] = ExpertTransition(
                source_expert_id=existing.source_expert_id,
                target_expert_id=existing.target_expert_id,
                count=existing.count + 1,
            )

    return transitions
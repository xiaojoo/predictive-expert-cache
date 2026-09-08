from __future__ import annotations

from dataclasses import dataclass

from .history import RoutingHistory


@dataclass(frozen=True, slots=True)
class ExpertAccessPattern:
    """
    Historical access pattern for one expert.
    """

    expert_id: int
    access_count: int
    first_seen_step: int
    last_seen_step: int
    layers: frozenset[int]

    def __post_init__(self) -> None:
        if self.expert_id < 0:
            raise ValueError("expert_id must be >= 0")

        if self.access_count <= 0:
            raise ValueError("access_count must be > 0")

        if self.first_seen_step < 0:
            raise ValueError("first_seen_step must be >= 0")

        if self.last_seen_step < self.first_seen_step:
            raise ValueError(
                "last_seen_step must be >= first_seen_step"
            )

        if not self.layers:
            raise ValueError("layers must not be empty")

        if any(layer_id < 0 for layer_id in self.layers):
            raise ValueError("layer_id must be >= 0")


def build_access_patterns(
    history: RoutingHistory,
) -> dict[int, ExpertAccessPattern]:
    """
    Build historical access patterns for every expert.
    """

    patterns: dict[int, ExpertAccessPattern] = {}

    for event in history.events():
        for expert_id in event.expert_ids:
            existing = patterns.get(expert_id)

            if existing is None:
                patterns[expert_id] = ExpertAccessPattern(
                    expert_id=expert_id,
                    access_count=1,
                    first_seen_step=event.step,
                    last_seen_step=event.step,
                    layers=frozenset({event.layer_id}),
                )
                continue

            patterns[expert_id] = ExpertAccessPattern(
                expert_id=expert_id,
                access_count=existing.access_count + 1,
                first_seen_step=existing.first_seen_step,
                last_seen_step=max(
                    existing.last_seen_step,
                    event.step,
                ),
                layers=existing.layers | frozenset(
                    {event.layer_id}
                ),
            )

    return patterns

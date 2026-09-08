from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RoutingEvent:
    """
    One MoE routing event.

    A routing event records which experts were selected for
    one token at one MoE layer and one inference step.
    """

    step: int
    token_id: int
    layer_id: int
    expert_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("step must be >= 0")

        if self.token_id < 0:
            raise ValueError("token_id must be >= 0")

        if self.layer_id < 0:
            raise ValueError("layer_id must be >= 0")

        if not self.expert_ids:
            raise ValueError("expert_ids must not be empty")

        if any(expert_id < 0 for expert_id in self.expert_ids):
            raise ValueError("expert_id must be >= 0")

        if len(set(self.expert_ids)) != len(self.expert_ids):
            raise ValueError("expert_ids must not contain duplicates")

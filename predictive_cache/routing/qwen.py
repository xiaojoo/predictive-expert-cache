from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .collector import RoutingEventCollector
from .models import RoutingEvent


def create_routing_event(
    *,
    step: int,
    token_id: int,
    layer_id: int,
    expert_ids: tuple[int, ...] | list[int],
) -> RoutingEvent:
    """Create a routing event from normalized Qwen router output."""
    return RoutingEvent(
        step=step,
        token_id=token_id,
        layer_id=layer_id,
        expert_ids=tuple(expert_ids),
    )


def collect_routing_event(
    collector: RoutingEventCollector,
    *,
    step: int,
    token_id: int,
    layer_id: int,
    expert_ids: tuple[int, ...] | list[int],
) -> RoutingEvent:
    """Create and collect one routing event."""
    event = create_routing_event(
        step=step,
        token_id=token_id,
        layer_id=layer_id,
        expert_ids=expert_ids,
    )
    collector.add(event)
    return event

def collect_routing_events(
    collector: RoutingEventCollector,
    *,
    step: int,
    token_id: int,
    layer_expert_ids: dict[int, tuple[int, ...] | list[int]],
) -> tuple[RoutingEvent, ...]:
    """Create and collect routing events for multiple MoE layers."""
    events: list[RoutingEvent] = []

    for layer_id, expert_ids in layer_expert_ids.items():
        event = collect_routing_event(
            collector,
            step=step,
            token_id=token_id,
            layer_id=layer_id,
            expert_ids=expert_ids,
        )
        events.append(event)

    return tuple(events)

@dataclass(frozen=True, slots=True)
class QwenRoutingOutput:
    """Normalized routing output for one inference step."""

    token_id: int
    layer_expert_ids: dict[int, tuple[int, ...]]

    def __post_init__(self) -> None:
        if self.token_id < 0:
            raise ValueError("token_id must be >= 0")

        normalized = {
            layer_id: tuple(expert_ids)
            for layer_id, expert_ids in self.layer_expert_ids.items()
        }

        if any(layer_id < 0 for layer_id in normalized):
            raise ValueError("layer_id must be >= 0")

        object.__setattr__(self, "layer_expert_ids", normalized)

def collect_qwen_routing_output(
    collector: RoutingEventCollector,
    *,
    step: int,
    output: QwenRoutingOutput,
) -> tuple[RoutingEvent, ...]:
    """Collect a normalized Qwen routing output."""
    return collect_routing_events(
        collector,
        step=step,
        token_id=output.token_id,
        layer_expert_ids=output.layer_expert_ids,
    )


class QwenRoutingAdapter(Protocol):
    """Convert model-specific router output into QwenRoutingOutput."""

    def adapt(
        self,
        *,
        token_id: int,
        layer_expert_ids: dict[int, tuple[int, ...] | list[int]],
    ) -> QwenRoutingOutput:
        ...


class DefaultQwenRoutingAdapter:
    """Default adapter for Qwen router output."""

    def adapt(
        self,
        *,
        token_id: int,
        layer_expert_ids: dict[int, tuple[int, ...] | list[int]],
    ) -> QwenRoutingOutput:
        return QwenRoutingOutput(
            token_id=token_id,
            layer_expert_ids=layer_expert_ids,
        )

    def adapt_router_output(
        self,
        *,
        token_id: int,
        token_position: int,
        layer_id: int,
        router_indices,
    ) -> QwenRoutingOutput:
        """Adapt one Qwen router output into normalized routing output."""
        if router_indices.ndim != 2:
            raise ValueError(
                "router_indices must have shape [seq_len, top_k]"
            )

        if token_position < 0 or token_position >= router_indices.shape[0]:
            raise IndexError("token_position out of range")

        expert_ids = tuple(
            int(expert_id)
            for expert_id in router_indices[token_position].tolist()
        )

        return QwenRoutingOutput(
            token_id=token_id,
            layer_expert_ids={
                layer_id: expert_ids,
            },
        )
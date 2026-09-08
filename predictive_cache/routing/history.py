from __future__ import annotations

from .models import RoutingEvent


class RoutingHistory:
    """
    Historical routing sequence.

    Stores routing events in inference order and provides
    lightweight history queries for later access-pattern analysis.
    """

    def __init__(
        self,
        events: tuple[RoutingEvent, ...] = (),
    ) -> None:
        self._events: list[RoutingEvent] = list(events)

    def add(self, event: RoutingEvent) -> None:
        self._events.append(event)

    def extend(self, events: list[RoutingEvent]) -> None:
        self._events.extend(events)

    def clear(self) -> None:
        self._events.clear()

    @property
    def size(self) -> int:
        return len(self._events)

    def events(self) -> tuple[RoutingEvent, ...]:
        return tuple(self._events)

    def recent(self, limit: int) -> tuple[RoutingEvent, ...]:
        if limit < 0:
            raise ValueError("limit must be >= 0")

        if limit == 0:
            return ()

        return tuple(self._events[-limit:])

    def by_step(self, step: int) -> tuple[RoutingEvent, ...]:
        return tuple(
            event
            for event in self._events
            if event.step == step
        )

    def by_layer(self, layer_id: int) -> tuple[RoutingEvent, ...]:
        return tuple(
            event
            for event in self._events
            if event.layer_id == layer_id
        )

    def by_expert(self, expert_id: int) -> tuple[RoutingEvent, ...]:
        if expert_id < 0:
            raise ValueError("expert_id must be >= 0")

        return tuple(
            event
            for event in self._events
            if expert_id in event.expert_ids
        )

    def expert_steps(self, expert_id: int) -> tuple[int, ...]:
        return tuple(
            event.step
            for event in self.by_expert(expert_id)
        )

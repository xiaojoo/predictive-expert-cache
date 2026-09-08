from __future__ import annotations

from .models import RoutingEvent


class RoutingEventCollector:
    """
    Collects routing events during inference.

    The collector keeps events in insertion order and provides
    lightweight query methods for later prediction analysis.
    """

    def __init__(self) -> None:
        self._events: list[RoutingEvent] = []

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

    def by_step_and_layer(
        self,
        step: int,
        layer_id: int,
    ) -> tuple[RoutingEvent, ...]:
        return tuple(
            event
            for event in self._events
            if event.step == step
            and event.layer_id == layer_id
        )

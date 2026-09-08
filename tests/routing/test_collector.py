from predictive_cache.routing import (
    RoutingEvent,
    RoutingEventCollector,
)


def make_event(
    step: int,
    token_id: int,
    layer_id: int,
    expert_ids: tuple[int, ...],
) -> RoutingEvent:
    return RoutingEvent(
        step=step,
        token_id=token_id,
        layer_id=layer_id,
        expert_ids=expert_ids,
    )


def test_collector_starts_empty() -> None:
    collector = RoutingEventCollector()

    assert collector.size == 0
    assert collector.events() == ()


def test_collector_adds_event() -> None:
    collector = RoutingEventCollector()

    event = make_event(
        step=0,
        token_id=100,
        layer_id=2,
        expert_ids=(3, 7),
    )

    collector.add(event)

    assert collector.size == 1
    assert collector.events() == (event,)


def test_collector_preserves_insertion_order() -> None:
    collector = RoutingEventCollector()

    event1 = make_event(0, 100, 1, (3, 7))
    event2 = make_event(1, 101, 1, (2, 8))
    event3 = make_event(2, 102, 1, (4, 9))

    collector.extend([event1, event2, event3])

    assert collector.events() == (
        event1,
        event2,
        event3,
    )


def test_collector_filters_by_step() -> None:
    collector = RoutingEventCollector()

    event1 = make_event(0, 100, 1, (3, 7))
    event2 = make_event(1, 101, 1, (2, 8))
    event3 = make_event(1, 102, 2, (4, 9))

    collector.extend([event1, event2, event3])

    assert collector.by_step(1) == (
        event2,
        event3,
    )


def test_collector_filters_by_layer() -> None:
    collector = RoutingEventCollector()

    event1 = make_event(0, 100, 1, (3, 7))
    event2 = make_event(1, 101, 2, (2, 8))
    event3 = make_event(2, 102, 1, (4, 9))

    collector.extend([event1, event2, event3])

    assert collector.by_layer(1) == (
        event1,
        event3,
    )


def test_collector_filters_by_step_and_layer() -> None:
    collector = RoutingEventCollector()

    event1 = make_event(0, 100, 1, (3, 7))
    event2 = make_event(1, 101, 1, (2, 8))
    event3 = make_event(1, 102, 2, (4, 9))

    collector.extend([event1, event2, event3])

    assert collector.by_step_and_layer(1, 1) == (
        event2,
    )


def test_collector_clear() -> None:
    collector = RoutingEventCollector()

    collector.add(
        make_event(
            step=0,
            token_id=100,
            layer_id=1,
            expert_ids=(3, 7),
        )
    )

    assert collector.size == 1

    collector.clear()

    assert collector.size == 0
    assert collector.events() == ()


def test_collector_returns_immutable_snapshot() -> None:
    collector = RoutingEventCollector()

    event = make_event(
        step=0,
        token_id=100,
        layer_id=1,
        expert_ids=(3, 7),
    )

    collector.add(event)

    events = collector.events()

    assert isinstance(events, tuple)
    assert events == (event,)


def test_collector_can_store_multiple_layers_per_step() -> None:
    collector = RoutingEventCollector()

    event1 = make_event(10, 1000, 4, (3, 17))
    event2 = make_event(10, 1000, 8, (7, 19))
    event3 = make_event(10, 1000, 12, (2, 41))

    collector.extend([event1, event2, event3])

    assert collector.by_step(10) == (
        event1,
        event2,
        event3,
    )

    assert collector.by_step_and_layer(10, 8) == (
        event2,
    )

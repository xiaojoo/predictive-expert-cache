import pytest

from predictive_cache.routing import (
    RoutingEvent,
    RoutingHistory,
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


def build_history() -> RoutingHistory:
    history = RoutingHistory()

    history.extend(
        [
            make_event(0, 100, 1, (3, 7)),
            make_event(1, 101, 1, (2, 8)),
            make_event(2, 102, 2, (3, 9)),
            make_event(3, 103, 1, (7, 19)),
            make_event(4, 104, 2, (3, 41)),
        ]
    )

    return history


def test_history_starts_empty() -> None:
    history = RoutingHistory()

    assert history.size == 0
    assert history.events() == ()


def test_history_adds_event() -> None:
    history = RoutingHistory()

    event = make_event(0, 100, 1, (3, 7))

    history.add(event)

    assert history.size == 1
    assert history.events() == (event,)


def test_history_preserves_event_order() -> None:
    history = build_history()

    events = history.events()

    assert [event.step for event in events] == [0, 1, 2, 3, 4]


def test_history_recent_returns_latest_events() -> None:
    history = build_history()

    recent = history.recent(2)

    assert [event.step for event in recent] == [3, 4]


def test_history_recent_larger_than_size() -> None:
    history = build_history()

    recent = history.recent(100)

    assert recent == history.events()


def test_history_recent_zero() -> None:
    history = build_history()

    assert history.recent(0) == ()


def test_history_recent_rejects_negative_limit() -> None:
    history = build_history()

    with pytest.raises(ValueError, match="limit must be >= 0"):
        history.recent(-1)


def test_history_filters_by_step() -> None:
    history = build_history()

    events = history.by_step(2)

    assert len(events) == 1
    assert events[0].step == 2
    assert events[0].expert_ids == (3, 9)


def test_history_filters_by_layer() -> None:
    history = build_history()

    events = history.by_layer(1)

    assert [event.step for event in events] == [0, 1, 3]


def test_history_filters_by_expert() -> None:
    history = build_history()

    events = history.by_expert(3)

    assert [event.step for event in events] == [0, 2, 4]


def test_history_rejects_negative_expert_id() -> None:
    history = build_history()

    with pytest.raises(ValueError, match="expert_id must be >= 0"):
        history.by_expert(-1)


def test_history_returns_expert_steps() -> None:
    history = build_history()

    assert history.expert_steps(7) == (0, 3)
    assert history.expert_steps(19) == (3,)
    assert history.expert_steps(99) == ()


def test_history_clear() -> None:
    history = build_history()

    assert history.size == 5

    history.clear()

    assert history.size == 0
    assert history.events() == ()


def test_history_can_be_initialized_with_events() -> None:
    events = (
        make_event(10, 1000, 4, (3, 17)),
        make_event(11, 1001, 4, (7, 19)),
    )

    history = RoutingHistory(events)

    assert history.size == 2
    assert history.events() == events

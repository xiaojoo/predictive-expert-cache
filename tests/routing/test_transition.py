import pytest

from predictive_cache.routing import (
    ExpertTransition,
    RoutingEvent,
    RoutingHistory,
    build_expert_transitions,
)


def make_history() -> RoutingHistory:
    history = RoutingHistory()

    history.extend(
        [
            RoutingEvent(
                step=0,
                token_id=100,
                layer_id=1,
                expert_ids=(3, 7),
            ),
            RoutingEvent(
                step=1,
                token_id=101,
                layer_id=1,
                expert_ids=(3, 7),
            ),
            RoutingEvent(
                step=2,
                token_id=102,
                layer_id=1,
                expert_ids=(3, 9),
            ),
            RoutingEvent(
                step=3,
                token_id=103,
                layer_id=1,
                expert_ids=(7, 19),
            ),
            RoutingEvent(
                step=4,
                token_id=104,
                layer_id=2,
                expert_ids=(3, 41, 12),
            ),
        ]
    )

    return history


def test_transition_creation() -> None:
    transition = ExpertTransition(
        source_expert_id=3,
        target_expert_id=7,
        count=3,
    )

    assert transition.source_expert_id == 3
    assert transition.target_expert_id == 7
    assert transition.count == 3


def test_transition_is_immutable() -> None:
    transition = ExpertTransition(
        source_expert_id=3,
        target_expert_id=7,
        count=1,
    )

    with pytest.raises(AttributeError):
        transition.count = 10  # type: ignore[misc]


def test_transition_rejects_invalid_source() -> None:
    with pytest.raises(
        ValueError,
        match="source_expert_id must be >= 0",
    ):
        ExpertTransition(
            source_expert_id=-1,
            target_expert_id=7,
            count=1,
        )


def test_transition_rejects_invalid_target() -> None:
    with pytest.raises(
        ValueError,
        match="target_expert_id must be >= 0",
    ):
        ExpertTransition(
            source_expert_id=3,
            target_expert_id=-1,
            count=1,
        )


def test_transition_rejects_invalid_count() -> None:
    with pytest.raises(
        ValueError,
        match="count must be > 0",
    ):
        ExpertTransition(
            source_expert_id=3,
            target_expert_id=7,
            count=0,
        )


def test_build_expert_transitions() -> None:
    transitions = build_expert_transitions(
        make_history()
    )

    assert set(transitions) == {
        (3, 7),
        (3, 9),
        (7, 19),
        (3, 41),
        (41, 12),
    }


def test_transition_counts() -> None:
    transitions = build_expert_transitions(
        make_history()
    )

    assert transitions[(3, 7)].count == 2
    assert transitions[(3, 9)].count == 1
    assert transitions[(7, 19)].count == 1
    assert transitions[(3, 41)].count == 1
    assert transitions[(41, 12)].count == 1


def test_single_expert_event_creates_no_transition() -> None:
    history = RoutingHistory()

    history.add(
        RoutingEvent(
            step=0,
            token_id=100,
            layer_id=1,
            expert_ids=(3,),
        )
    )

    transitions = build_expert_transitions(history)

    assert transitions == {}


def test_empty_history() -> None:
    history = RoutingHistory()

    transitions = build_expert_transitions(history)

    assert transitions == {}


def test_transitions_do_not_mutate_history() -> None:
    history = make_history()
    before = history.events()

    build_expert_transitions(history)

    assert history.events() == before
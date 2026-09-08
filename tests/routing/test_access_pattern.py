import pytest

from predictive_cache.routing import (
    ExpertAccessPattern,
    RoutingEvent,
    RoutingHistory,
    build_access_patterns,
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
                expert_ids=(2, 8),
            ),
            RoutingEvent(
                step=2,
                token_id=102,
                layer_id=2,
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
                expert_ids=(3, 41),
            ),
        ]
    )

    return history


def test_access_pattern_creation() -> None:
    pattern = ExpertAccessPattern(
        expert_id=3,
        access_count=3,
        first_seen_step=0,
        last_seen_step=4,
        layers=frozenset({1, 2}),
    )

    assert pattern.expert_id == 3
    assert pattern.access_count == 3
    assert pattern.first_seen_step == 0
    assert pattern.last_seen_step == 4
    assert pattern.layers == frozenset({1, 2})


def test_access_pattern_is_immutable() -> None:
    pattern = ExpertAccessPattern(
        expert_id=3,
        access_count=1,
        first_seen_step=0,
        last_seen_step=0,
        layers=frozenset({1}),
    )

    with pytest.raises(AttributeError):
        pattern.access_count = 10  # type: ignore[misc]


def test_access_pattern_rejects_invalid_expert_id() -> None:
    with pytest.raises(ValueError, match="expert_id must be >= 0"):
        ExpertAccessPattern(
            expert_id=-1,
            access_count=1,
            first_seen_step=0,
            last_seen_step=0,
            layers=frozenset({1}),
        )


def test_access_pattern_rejects_invalid_access_count() -> None:
    with pytest.raises(ValueError, match="access_count must be > 0"):
        ExpertAccessPattern(
            expert_id=1,
            access_count=0,
            first_seen_step=0,
            last_seen_step=0,
            layers=frozenset({1}),
        )


def test_access_pattern_rejects_invalid_steps() -> None:
    with pytest.raises(ValueError, match="first_seen_step"):
        ExpertAccessPattern(
            expert_id=1,
            access_count=1,
            first_seen_step=5,
            last_seen_step=4,
            layers=frozenset({1}),
        )


def test_access_pattern_requires_layers() -> None:
    with pytest.raises(ValueError, match="layers must not be empty"):
        ExpertAccessPattern(
            expert_id=1,
            access_count=1,
            first_seen_step=0,
            last_seen_step=0,
            layers=frozenset(),
        )


def test_build_access_patterns() -> None:
    patterns = build_access_patterns(make_history())

    assert set(patterns) == {
        2,
        3,
        7,
        8,
        9,
        19,
        41,
    }


def test_build_access_patterns_for_expert_3() -> None:
    patterns = build_access_patterns(make_history())

    pattern = patterns[3]

    assert pattern.access_count == 3
    assert pattern.first_seen_step == 0
    assert pattern.last_seen_step == 4
    assert pattern.layers == frozenset({1, 2})


def test_build_access_patterns_for_expert_7() -> None:
    patterns = build_access_patterns(make_history())

    pattern = patterns[7]

    assert pattern.access_count == 2
    assert pattern.first_seen_step == 0
    assert pattern.last_seen_step == 3
    assert pattern.layers == frozenset({1})


def test_build_access_patterns_for_single_access_expert() -> None:
    patterns = build_access_patterns(make_history())

    pattern = patterns[19]

    assert pattern.access_count == 1
    assert pattern.first_seen_step == 3
    assert pattern.last_seen_step == 3
    assert pattern.layers == frozenset({1})


def test_build_access_patterns_empty_history() -> None:
    history = RoutingHistory()

    patterns = build_access_patterns(history)

    assert patterns == {}


def test_access_patterns_do_not_mutate_history() -> None:
    history = make_history()
    before = history.events()

    build_access_patterns(history)

    assert history.events() == before

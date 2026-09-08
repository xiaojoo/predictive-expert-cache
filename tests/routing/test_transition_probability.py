import pytest

from predictive_cache.routing import (
    ExpertTransition,
    ExpertTransitionProbability,
    build_transition_probabilities,
)


def make_transitions() -> dict[
    tuple[int, int],
    ExpertTransition,
]:
    return {
        (3, 7): ExpertTransition(
            source_expert_id=3,
            target_expert_id=7,
            count=10,
        ),
        (3, 9): ExpertTransition(
            source_expert_id=3,
            target_expert_id=9,
            count=5,
        ),
        (3, 41): ExpertTransition(
            source_expert_id=3,
            target_expert_id=41,
            count=5,
        ),
        (7, 19): ExpertTransition(
            source_expert_id=7,
            target_expert_id=19,
            count=4,
        ),
    }


def test_transition_probability_creation() -> None:
    probability = ExpertTransitionProbability(
        source_expert_id=3,
        target_expert_id=7,
        probability=0.5,
    )

    assert probability.source_expert_id == 3
    assert probability.target_expert_id == 7
    assert probability.probability == 0.5


def test_transition_probability_is_immutable() -> None:
    probability = ExpertTransitionProbability(
        source_expert_id=3,
        target_expert_id=7,
        probability=0.5,
    )

    with pytest.raises(AttributeError):
        probability.probability = 0.8  # type: ignore[misc]


def test_transition_probability_rejects_invalid_source() -> None:
    with pytest.raises(
        ValueError,
        match="source_expert_id must be >= 0",
    ):
        ExpertTransitionProbability(
            source_expert_id=-1,
            target_expert_id=7,
            probability=0.5,
        )


def test_transition_probability_rejects_invalid_target() -> None:
    with pytest.raises(
        ValueError,
        match="target_expert_id must be >= 0",
    ):
        ExpertTransitionProbability(
            source_expert_id=3,
            target_expert_id=-1,
            probability=0.5,
        )


def test_transition_probability_rejects_invalid_probability() -> None:
    with pytest.raises(
        ValueError,
        match="probability must be between 0 and 1",
    ):
        ExpertTransitionProbability(
            source_expert_id=3,
            target_expert_id=7,
            probability=1.1,
        )


def test_build_transition_probabilities() -> None:
    probabilities = build_transition_probabilities(
        make_transitions()
    )

    assert set(probabilities) == {
        (3, 7),
        (3, 9),
        (3, 41),
        (7, 19),
    }


def test_probabilities_are_normalized_per_source() -> None:
    probabilities = build_transition_probabilities(
        make_transitions()
    )

    assert probabilities[(3, 7)].probability == pytest.approx(
        0.5
    )

    assert probabilities[(3, 9)].probability == pytest.approx(
        0.25
    )

    assert probabilities[(3, 41)].probability == pytest.approx(
        0.25
    )


def test_probabilities_sum_to_one_per_source() -> None:
    probabilities = build_transition_probabilities(
        make_transitions()
    )

    source_3_total = sum(
        item.probability
        for item in probabilities.values()
        if item.source_expert_id == 3
    )

    source_7_total = sum(
        item.probability
        for item in probabilities.values()
        if item.source_expert_id == 7
    )

    assert source_3_total == pytest.approx(1.0)
    assert source_7_total == pytest.approx(1.0)


def test_single_transition_has_probability_one() -> None:
    transitions = {
        (3, 7): ExpertTransition(
            source_expert_id=3,
            target_expert_id=7,
            count=8,
        ),
    }

    probabilities = build_transition_probabilities(
        transitions
    )

    assert probabilities[(3, 7)].probability == pytest.approx(
        1.0
    )


def test_empty_transitions() -> None:
    probabilities = build_transition_probabilities({})

    assert probabilities == {}


def test_transition_counts_are_not_mutated() -> None:
    transitions = make_transitions()

    before = dict(transitions)

    build_transition_probabilities(transitions)

    assert transitions == before
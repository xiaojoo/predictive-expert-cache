import math

import pytest

from predictive_cache.routing.normalization import (
    NormalizedExpertFeatures,
)
from predictive_cache.routing.scoring import (
    PredictionScore,
    calculate_prediction_score,
    recency_score,
)


def make_features(
    expert_id: int,
    frequency_score: float,
    recency: int,
    transition_probability: float,
) -> NormalizedExpertFeatures:
    return NormalizedExpertFeatures(
        expert_id=expert_id,
        frequency_score=frequency_score,
        recency=recency,
        transition_probability=transition_probability,
    )


def test_recency_score_is_one_for_zero_recency():
    score = recency_score(
        recency=0,
        recency_decay=8.0,
    )

    assert score == 1.0


def test_recency_score_follows_exponential_decay():
    score = recency_score(
        recency=8,
        recency_decay=8.0,
    )

    assert score == pytest.approx(math.exp(-1.0))


def test_recency_score_decreases_as_recency_increases():
    score_0 = recency_score(
        recency=0,
        recency_decay=8.0,
    )
    score_8 = recency_score(
        recency=8,
        recency_decay=8.0,
    )
    score_16 = recency_score(
        recency=16,
        recency_decay=8.0,
    )

    assert score_0 > score_8 > score_16


def test_recency_score_is_between_zero_and_one():
    for recency in (0, 1, 8, 16, 100):
        score = recency_score(
            recency=recency,
            recency_decay=8.0,
        )

        assert 0.0 <= score <= 1.0


def test_negative_recency_is_rejected():
    with pytest.raises(ValueError):
        recency_score(
            recency=-1,
            recency_decay=8.0,
        )


def test_invalid_recency_decay_is_rejected():
    with pytest.raises(ValueError):
        recency_score(
            recency=1,
            recency_decay=0.0,
        )


def test_prediction_score_combines_all_features():
    features = make_features(
        expert_id=7,
        frequency_score=0.8,
        recency=0,
        transition_probability=0.6,
    )

    result = calculate_prediction_score(
        features,
        frequency_weight=0.30,
        recency_weight=0.20,
        transition_weight=0.50,
        recency_decay=8.0,
    )

    expected = (
        0.30 * 0.8
        + 0.20 * 1.0
        + 0.50 * 0.6
    )

    assert result.score == pytest.approx(expected)


def test_prediction_score_uses_recency_decay():
    features = make_features(
        expert_id=7,
        frequency_score=0.0,
        recency=8,
        transition_probability=0.0,
    )

    result = calculate_prediction_score(
        features,
        frequency_weight=0.30,
        recency_weight=0.20,
        transition_weight=0.50,
        recency_decay=8.0,
    )

    expected = 0.20 * math.exp(-1.0)

    assert result.score == pytest.approx(expected)


def test_prediction_score_preserves_expert_id():
    features = make_features(
        expert_id=42,
        frequency_score=0.5,
        recency=2,
        transition_probability=0.7,
    )

    result = calculate_prediction_score(
        features,
        frequency_weight=0.30,
        recency_weight=0.20,
        transition_weight=0.50,
        recency_decay=8.0,
    )

    assert result.expert_id == 42


def test_prediction_score_contains_component_scores():
    features = make_features(
        expert_id=3,
        frequency_score=0.7,
        recency=4,
        transition_probability=0.9,
    )

    result = calculate_prediction_score(
        features,
        frequency_weight=0.30,
        recency_weight=0.20,
        transition_weight=0.50,
        recency_decay=8.0,
    )

    assert result.frequency_score == 0.7
    assert result.transition_score == 0.9
    assert result.recency_score == pytest.approx(
        math.exp(-4 / 8.0)
    )


def test_prediction_score_is_bounded_between_zero_and_one():
    features = make_features(
        expert_id=1,
        frequency_score=1.0,
        recency=0,
        transition_probability=1.0,
    )

    result = calculate_prediction_score(
        features,
        frequency_weight=0.30,
        recency_weight=0.20,
        transition_weight=0.50,
        recency_decay=8.0,
    )

    assert 0.0 <= result.score <= 1.0
    assert result.score == pytest.approx(1.0)


def test_zero_features_produce_zero_score():
    features = make_features(
        expert_id=1,
        frequency_score=0.0,
        recency=100,
        transition_probability=0.0,
    )

    result = calculate_prediction_score(
        features,
        frequency_weight=0.30,
        recency_weight=0.20,
        transition_weight=0.50,
        recency_decay=8.0,
    )

    assert result.score > 0.0
    assert result.score < 0.20


def test_invalid_weights_are_rejected():
    features = make_features(
        expert_id=1,
        frequency_score=0.5,
        recency=1,
        transition_probability=0.5,
    )

    with pytest.raises(ValueError):
        calculate_prediction_score(
            features,
            frequency_weight=-0.1,
            recency_weight=0.2,
            transition_weight=0.5,
            recency_decay=8.0,
        )


def test_zero_total_weights_are_rejected():
    features = make_features(
        expert_id=1,
        frequency_score=0.5,
        recency=1,
        transition_probability=0.5,
    )

    with pytest.raises(ValueError):
        calculate_prediction_score(
            features,
            frequency_weight=0.0,
            recency_weight=0.0,
            transition_weight=0.0,
            recency_decay=8.0,
        )


def test_prediction_score_has_expected_type():
    features = make_features(
        expert_id=5,
        frequency_score=0.4,
        recency=3,
        transition_probability=0.8,
    )

    result = calculate_prediction_score(
        features,
        frequency_weight=0.30,
        recency_weight=0.20,
        transition_weight=0.50,
        recency_decay=8.0,
    )

    assert isinstance(result, PredictionScore)

def test_prediction_score_increases_when_frequency_improves():
    low_frequency = make_features(
        expert_id=7,
        frequency_score=0.2,
        recency=4,
        transition_probability=0.3,
    )

    high_frequency = make_features(
        expert_id=7,
        frequency_score=0.8,
        recency=4,
        transition_probability=0.3,
    )

    low_result = calculate_prediction_score(
        low_frequency,
        frequency_weight=0.5,
        recency_weight=0.2,
        transition_weight=0.3,
        recency_decay=8.0,
    )

    high_result = calculate_prediction_score(
        high_frequency,
        frequency_weight=0.5,
        recency_weight=0.2,
        transition_weight=0.3,
        recency_decay=8.0,
    )

    assert high_result.score > low_result.score


def test_prediction_score_increases_when_transition_probability_improves():
    low_transition = make_features(
        expert_id=7,
        frequency_score=0.4,
        recency=4,
        transition_probability=0.1,
    )

    high_transition = make_features(
        expert_id=7,
        frequency_score=0.4,
        recency=4,
        transition_probability=0.9,
    )

    low_result = calculate_prediction_score(
        low_transition,
        frequency_weight=0.3,
        recency_weight=0.2,
        transition_weight=0.5,
        recency_decay=8.0,
    )

    high_result = calculate_prediction_score(
        high_transition,
        frequency_weight=0.3,
        recency_weight=0.2,
        transition_weight=0.5,
        recency_decay=8.0,
    )

    assert high_result.score > low_result.score


def test_prediction_score_components_are_preserved():
    features = make_features(
        expert_id=17,
        frequency_score=0.8,
        recency=0,
        transition_probability=0.6,
    )

    result = calculate_prediction_score(
        features,
        frequency_weight=0.3,
        recency_weight=0.2,
        transition_weight=0.5,
        recency_decay=8.0,
    )

    assert result.expert_id == 17
    assert result.frequency_score == 0.8
    assert result.recency_score == 1.0
    assert result.transition_score == 0.6
    assert 0.0 <= result.score <= 1.0

def test_prediction_scores_can_be_ordered_by_score() -> None:
    low = calculate_prediction_score(
        make_features(
            expert_id=7,
            frequency_score=0.2,
            recency=5,
            transition_probability=0.1,
        ),
        frequency_weight=1.0,
        recency_weight=0.0,
        transition_weight=0.0,
        recency_decay=10.0,
    )

    high = calculate_prediction_score(
        make_features(
            expert_id=3,
            frequency_score=0.8,
            recency=5,
            transition_probability=0.1,
        ),
        frequency_weight=1.0,
        recency_weight=0.0,
        transition_weight=0.0,
        recency_decay=10.0,
    )

    ordered = sorted(
        (low, high),
        key=lambda prediction: prediction.score,
        reverse=True,
    )

    assert [prediction.expert_id for prediction in ordered] == [3, 7]
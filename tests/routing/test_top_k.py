import pytest

from predictive_cache.routing.scoring import PredictionScore
from predictive_cache.routing.top_k import (
    select_top_k_predictions,
)


def make_prediction(
    expert_id: int,
    score: float,
) -> PredictionScore:
    return PredictionScore(
        expert_id=expert_id,
        score=score,
        frequency_score=score,
        recency_score=score,
        transition_score=score,
    )


def test_select_top_k_returns_highest_scores_first():
    predictions = {
        0: make_prediction(0, 0.31),
        1: make_prediction(1, 0.87),
        2: make_prediction(2, 0.44),
        3: make_prediction(3, 0.72),
        4: make_prediction(4, 0.16),
        5: make_prediction(5, 0.65),
    }

    result = select_top_k_predictions(
        predictions,
        top_k=3,
        min_score=0.0,
    )

    assert [prediction.expert_id for prediction in result] == [
        1,
        3,
        5,
    ]

    assert [prediction.score for prediction in result] == [
        0.87,
        0.72,
        0.65,
    ]


def test_select_top_k_returns_at_most_k_predictions():
    predictions = {
        0: make_prediction(0, 0.9),
        1: make_prediction(1, 0.8),
        2: make_prediction(2, 0.7),
        3: make_prediction(3, 0.6),
    }

    result = select_top_k_predictions(
        predictions,
        top_k=2,
        min_score=0.0,
    )

    assert len(result) == 2


def test_top_k_larger_than_candidates_returns_all():
    predictions = {
        0: make_prediction(0, 0.5),
        1: make_prediction(1, 0.8),
    }

    result = select_top_k_predictions(
        predictions,
        top_k=10,
        min_score=0.0,
    )

    assert [prediction.expert_id for prediction in result] == [
        1,
        0,
    ]


def test_top_k_one_returns_best_prediction():
    predictions = {
        0: make_prediction(0, 0.4),
        1: make_prediction(1, 0.9),
        2: make_prediction(2, 0.7),
    }

    result = select_top_k_predictions(
        predictions,
        top_k=1,
        min_score=0.0,
    )

    assert len(result) == 1
    assert result[0].expert_id == 1
    assert result[0].score == pytest.approx(0.9)


def test_min_score_filters_low_predictions():
    predictions = {
        0: make_prediction(0, 0.2),
        1: make_prediction(1, 0.8),
        2: make_prediction(2, 0.6),
        3: make_prediction(3, 0.3),
    }

    result = select_top_k_predictions(
        predictions,
        top_k=10,
        min_score=0.5,
    )

    assert [prediction.expert_id for prediction in result] == [
        1,
        2,
    ]


def test_min_score_is_inclusive():
    predictions = {
        0: make_prediction(0, 0.5),
        1: make_prediction(1, 0.8),
    }

    result = select_top_k_predictions(
        predictions,
        top_k=10,
        min_score=0.5,
    )

    assert [prediction.expert_id for prediction in result] == [
        1,
        0,
    ]


def test_all_predictions_below_min_score_returns_empty():
    predictions = {
        0: make_prediction(0, 0.2),
        1: make_prediction(1, 0.3),
    }

    result = select_top_k_predictions(
        predictions,
        top_k=10,
        min_score=0.5,
    )

    assert result == ()


def test_empty_predictions_returns_empty():
    result = select_top_k_predictions(
        {},
        top_k=4,
        min_score=0.0,
    )

    assert result == ()


def test_invalid_top_k_is_rejected():
    predictions = {
        0: make_prediction(0, 0.5),
    }

    with pytest.raises(ValueError):
        select_top_k_predictions(
            predictions,
            top_k=0,
            min_score=0.0,
        )


def test_negative_top_k_is_rejected():
    predictions = {
        0: make_prediction(0, 0.5),
    }

    with pytest.raises(ValueError):
        select_top_k_predictions(
            predictions,
            top_k=-1,
            min_score=0.0,
        )


def test_invalid_min_score_is_rejected():
    predictions = {
        0: make_prediction(0, 0.5),
    }

    with pytest.raises(ValueError):
        select_top_k_predictions(
            predictions,
            top_k=1,
            min_score=-0.1,
        )

    with pytest.raises(ValueError):
        select_top_k_predictions(
            predictions,
            top_k=1,
            min_score=1.1,
        )


def test_equal_scores_have_deterministic_order():
    predictions = {
        5: make_prediction(5, 0.8),
        2: make_prediction(2, 0.8),
        9: make_prediction(9, 0.8),
    }

    result = select_top_k_predictions(
        predictions,
        top_k=3,
        min_score=0.0,
    )

    assert [prediction.expert_id for prediction in result] == [
        2,
        5,
        9,
    ]


def test_original_predictions_are_not_modified():
    predictions = {
        0: make_prediction(0, 0.4),
        1: make_prediction(1, 0.9),
        2: make_prediction(2, 0.7),
    }

    original = dict(predictions)

    select_top_k_predictions(
        predictions,
        top_k=2,
        min_score=0.0,
    )

    assert predictions == original


def test_result_contains_original_prediction_objects():
    prediction = make_prediction(7, 0.95)

    predictions = {
        7: prediction,
    }

    result = select_top_k_predictions(
        predictions,
        top_k=1,
        min_score=0.0,
    )

    assert result[0] is prediction

def test_prediction_scores_select_highest_scoring_experts() -> None:
    predictions = {
        1: PredictionScore(
            expert_id=1,
            score=0.25,
            frequency_score=0.2,
            recency_score=0.3,
            transition_score=0.2,
        ),
        2: PredictionScore(
            expert_id=2,
            score=0.90,
            frequency_score=0.9,
            recency_score=0.9,
            transition_score=0.9,
        ),
        3: PredictionScore(
            expert_id=3,
            score=0.65,
            frequency_score=0.6,
            recency_score=0.7,
            transition_score=0.6,
        ),
    }

    selected = select_top_k_predictions(
        predictions,
        top_k=2,
        min_score=0.0,
    )

    assert [prediction.expert_id for prediction in selected] == [2, 3]
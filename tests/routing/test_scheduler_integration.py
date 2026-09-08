from predictive_cache.routing.scoring import PredictionScore
from predictive_cache.routing.top_k import select_top_k_predictions


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


def test_scheduler_prediction_pipeline_produces_top_k_candidates():
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


def test_scheduler_prediction_pipeline_respects_min_score():
    predictions = {
        0: make_prediction(0, 0.20),
        1: make_prediction(1, 0.80),
        2: make_prediction(2, 0.60),
        3: make_prediction(3, 0.30),
    }

    result = select_top_k_predictions(
        predictions,
        top_k=4,
        min_score=0.5,
    )

    assert [prediction.expert_id for prediction in result] == [
        1,
        2,
    ]


def test_scheduler_prediction_pipeline_preserves_prediction_score():
    prediction = make_prediction(7, 0.95)

    result = select_top_k_predictions(
        {7: prediction},
        top_k=1,
        min_score=0.0,
    )

    assert result[0].expert_id == 7
    assert result[0].score == 0.95
    assert result[0].frequency_score == 0.95
    assert result[0].recency_score == 0.95
    assert result[0].transition_score == 0.95
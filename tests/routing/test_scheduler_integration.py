from predictive_cache.predictive_scheduler import (
    CapacityManager,
    CostModel,
    ExpertPrediction,
    SchedulerAction,
    SchedulerPolicy,
    TransferCost,
)
from predictive_cache.predictive_scheduler.adapter import (
    prediction_score_to_scheduler_prediction,
    prediction_scores_to_scheduler_predictions,
)
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


def test_prediction_score_converts_to_scheduler_prediction():
    prediction = make_prediction(17, 0.90)

    result = prediction_score_to_scheduler_prediction(
        prediction,
        expert_size_mb=256.0,
    )

    assert isinstance(result, ExpertPrediction)
    assert result.expert_id == 17
    assert result.probability == 0.90
    assert result.expert_size_mb == 256.0


def test_prediction_score_batch_conversion_preserves_top_k_order():
    predictions = {
        1: make_prediction(1, 0.40),
        2: make_prediction(2, 0.90),
        3: make_prediction(3, 0.70),
    }

    top_k = select_top_k_predictions(
        predictions,
        top_k=2,
        min_score=0.0,
    )

    result = prediction_scores_to_scheduler_predictions(
        top_k,
        expert_sizes_mb={
            1: 128.0,
            2: 256.0,
            3: 512.0,
        },
    )

    assert [prediction.expert_id for prediction in result] == [
        2,
        3,
    ]

    assert [prediction.probability for prediction in result] == [
        0.90,
        0.70,
    ]

    assert [prediction.expert_size_mb for prediction in result] == [
        256.0,
        512.0,
    ]


def test_top_k_predictions_can_drive_scheduler_policy():
    predictions = {
        17: make_prediction(17, 0.90),
        81: make_prediction(81, 0.20),
    }

    top_k = select_top_k_predictions(
        predictions,
        top_k=2,
        min_score=0.0,
    )

    scheduler_predictions = (
        prediction_scores_to_scheduler_predictions(
            top_k,
            expert_sizes_mb={
                17: 256.0,
                81: 256.0,
            },
        )
    )

    policy = SchedulerPolicy(
        cost_model=CostModel(
            default_reuse_cost_ms=50.0,
        ),
    )

    transfer_cost = TransferCost(
        gpu_to_ram_ms=5.0,
        ram_to_gpu_ms=10.0,
    )

    cache = CapacityManager(
        capacity_mb=4096.0,
        used_mb=1024.0,
    )

    decisions = [
        policy.decide(
            prediction,
            transfer_cost,
            cache,
        )
        for prediction in scheduler_predictions
    ]

    assert decisions[0].expert_id == 17
    assert decisions[0].action is SchedulerAction.PREFETCH

    assert decisions[1].expert_id == 81
    assert decisions[1].action is SchedulerAction.WAIT
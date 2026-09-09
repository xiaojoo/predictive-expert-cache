from predictive_cache.scheduler import (
    ExpertScheduler,
    PrefetchRequest,
)
from predictive_cache.predictive_scheduler import (
    TransferCost,
)
from predictive_cache.types import ExpertPrediction
import pytest

class FakeCache:
    def __init__(self, cached=()):
        self._cached = set(cached)

    def cached_experts(self):
        return list(self._cached)


def prediction(
    expert_id: int,
    score: float,
) -> ExpertPrediction:
    return ExpertPrediction(
        expert_id=expert_id,
        score=score,
    )


def test_scheduler_deduplicates_predictions():
    cache = FakeCache()

    scheduler = ExpertScheduler(
        cache,
        default_expert_size_mb=100,
    )

    result = scheduler.plan_predictions(
        [
            prediction(17, 0.7),
            prediction(17, 0.9),
            prediction(32, 0.8),
        ],
        expert_sizes_mb={
            17: 100,
            32: 100,
        },
        capacity_mb=1000,
        used_mb=0,
        reuse_cost_ms=1000,
    )

    assert [item.expert_id for item in result] == [
        17,
        32,
    ]

    assert result[0].priority > result[1].priority


def test_scheduler_skips_cached_experts():
    cache = FakeCache(
        cached=[17],
    )

    scheduler = ExpertScheduler(
        cache,
        default_expert_size_mb=100,
    )

    result = scheduler.plan_predictions(
        [
            prediction(17, 0.99),
            prediction(32, 0.80),
        ],
        expert_sizes_mb={
            17: 100,
            32: 100,
        },
        capacity_mb=1000,
        used_mb=0,
        reuse_cost_ms=1000,
    )

    assert [item.expert_id for item in result] == [32]


def test_scheduler_respects_capacity():
    cache = FakeCache()

    scheduler = ExpertScheduler(
        cache,
    )

    result = scheduler.plan_predictions(
        [
            prediction(17, 0.9),
            prediction(32, 0.8),
            prediction(81, 0.7),
        ],
        expert_sizes_mb={
            17: 600,
            32: 600,
            81: 200,
        },
        capacity_mb=1000,
        used_mb=0,
        reuse_cost_ms=1000,
    )

    assert [item.expert_id for item in result] == [
        17,
        81,
    ]


def test_scheduler_ranks_by_benefit():
    cache = FakeCache()

    scheduler = ExpertScheduler(
        cache,
    )

    result = scheduler.plan_predictions(
        [
            prediction(17, 0.7),
            prediction(32, 0.9),
        ],
        expert_sizes_mb={
            17: 100,
            32: 100,
        },
        transfer_costs={
            17: TransferCost(
                gpu_to_ram_ms=0,
                ram_to_gpu_ms=10,
            ),
            32: TransferCost(
                gpu_to_ram_ms=0,
                ram_to_gpu_ms=50,
            ),
        },
        capacity_mb=1000,
        used_mb=0,
        reuse_cost_ms=100,
    )

    assert [item.expert_id for item in result] == [
        17,
        32,
    ]

    assert result[0].priority > result[1].priority

def test_plan_predictions_uses_distance_for_priority(
    cache,
):
    scheduler = ExpertScheduler(cache)

    predictions = [
        ExpertPrediction(
            expert_id=1,
            score=0.8,
            estimated_distance=1.0,
        ),
        ExpertPrediction(
            expert_id=2,
            score=0.9,
            estimated_distance=9.0,
        ),
    ]

    requests = scheduler.plan_predictions(
        predictions,
        cached_experts=[],
    )

    assert [request.expert_id for request in requests] == [
        1,
        2,
    ]

    assert requests[0].score == pytest.approx(0.8)
    assert requests[0].priority == pytest.approx(0.4)

    assert requests[1].score == pytest.approx(0.9)
    assert requests[1].priority == pytest.approx(0.09)


def test_plan_predictions_without_distance_keeps_priority(
    cache,
):
    scheduler = ExpertScheduler(cache)

    predictions = [
        ExpertPrediction(
            expert_id=1,
            score=0.8,
            estimated_distance=None,
        ),
    ]

    requests = scheduler.plan_predictions(
        predictions,
        cached_experts=[],
    )

    assert len(requests) == 1

    assert requests[0].score == pytest.approx(0.8)
    assert requests[0].priority == pytest.approx(0.8)
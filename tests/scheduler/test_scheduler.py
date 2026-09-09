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

def test_distance_factor_none():
    assert ExpertScheduler._distance_factor(None) == 1.0


def test_distance_factor_zero():
    assert ExpertScheduler._distance_factor(0.0) == 1.0


def test_distance_factor_one():
    assert ExpertScheduler._distance_factor(1.0) == 0.5


def test_distance_factor_three():
    assert ExpertScheduler._distance_factor(3.0) == 0.25


def test_distance_factor_nine():
    assert ExpertScheduler._distance_factor(9.0) == 0.1


def test_distance_factor_rejects_negative():
    with pytest.raises(ValueError, match="distance must be >= 0"):
        ExpertScheduler._distance_factor(-1.0)

def test_distance_changes_priority_but_not_score(cache):
    scheduler = ExpertScheduler(cache)

    predictions = [
        ExpertPrediction(
            expert_id=1,
            score=0.8,
            estimated_distance=1.0,
        ),
        ExpertPrediction(
            expert_id=2,
            score=0.8,
            estimated_distance=9.0,
        ),
    ]

    requests = scheduler.plan_predictions(
        predictions,
        cached_experts=[],
    )

    assert requests[0].expert_id == 1
    assert requests[1].expert_id == 2

    assert requests[0].score == pytest.approx(0.8)
    assert requests[1].score == pytest.approx(0.8)

    assert requests[0].priority == pytest.approx(0.4)
    assert requests[1].priority == pytest.approx(0.08)

def test_distance_changes_capacity_aware_priority_but_not_score(cache):
    scheduler = ExpertScheduler(cache)

    predictions = [
        ExpertPrediction(
            expert_id=1,
            score=0.8,
            estimated_distance=1.0,
        ),
        ExpertPrediction(
            expert_id=2,
            score=0.8,
            estimated_distance=9.0,
        ),
    ]

    requests = scheduler.plan_predictions(
        predictions,
        cached_experts=[],
        expert_sizes_mb={
            1: 1,
            2: 1,
        },
        capacity_mb=100,
        used_mb=0,
        reuse_cost_ms=100,
    )

    assert requests[0].expert_id == 1
    assert requests[1].expert_id == 2

    assert requests[0].score == pytest.approx(80.0)
    assert requests[1].score == pytest.approx(80.0)

    assert requests[0].priority == pytest.approx(40.0)
    assert requests[1].priority == pytest.approx(8.0)

def test_distance_does_not_override_minimum_benefit(cache):
    scheduler = ExpertScheduler(
        cache,
        minimum_benefit=10.0,
    )

    predictions = [
        ExpertPrediction(
            expert_id=1,
            score=0.09,
            estimated_distance=0.0,
        ),
        ExpertPrediction(
            expert_id=2,
            score=0.2,
            estimated_distance=9.0,
        ),
    ]

    requests = scheduler.plan_predictions(
        predictions,
        cached_experts=[],
        reuse_cost_ms=1.0,
    )

    assert [request.expert_id for request in requests] == []

def test_distance_does_not_override_minimum_benefit(cache):
    scheduler = ExpertScheduler(
        cache,
        minimum_benefit=10.0,
    )

    predictions = [
        ExpertPrediction(
            expert_id=1,
            score=0.09,
            estimated_distance=0.0,
        ),
        ExpertPrediction(
            expert_id=2,
            score=0.2,
            estimated_distance=9.0,
        ),
    ]

    requests = scheduler.plan_predictions(
        predictions,
        cached_experts=[],
        reuse_cost_ms=100.0,
    )

    assert [request.expert_id for request in requests] == [2]

    assert requests[0].score == pytest.approx(20.0)
    assert requests[0].priority == pytest.approx(2.0)
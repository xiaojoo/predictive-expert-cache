from __future__ import annotations

import pytest

from predictive_cache.predictive_scheduler.capacity import CapacityManager
from predictive_cache.predictive_scheduler.cost_model import CostModel
from predictive_cache.predictive_scheduler.models import (
    ExpertPrediction,
    SchedulerAction,
    TransferCost,
)
from predictive_cache.predictive_scheduler.policy import SchedulerPolicy


def _prediction(
    expert_id: int = 1,
    probability: float = 0.9,
    size_mb: float = 100.0,
) -> ExpertPrediction:
    return ExpertPrediction(
        expert_id=expert_id,
        probability=probability,
        expert_size_mb=size_mb,
    )


def test_roi_positive_when_expected_reuse_exceeds_transfer() -> None:
    model = CostModel(default_reuse_cost_ms=20.0)
    prediction = _prediction(probability=0.9)
    transfer = TransferCost(gpu_to_ram_ms=0.0, ram_to_gpu_ms=2.0)

    benefit = model.benefit(prediction, transfer)

    assert benefit == pytest.approx(16.0)


def test_roi_negative_when_transfer_cost_exceeds_expected_reuse() -> None:
    model = CostModel(default_reuse_cost_ms=2.0)
    prediction = _prediction(probability=0.5)
    transfer = TransferCost(gpu_to_ram_ms=0.0, ram_to_gpu_ms=3.0)

    benefit = model.benefit(prediction, transfer)

    assert benefit == pytest.approx(-2.0)


def test_eviction_cost_reduces_roi() -> None:
    model = CostModel(default_reuse_cost_ms=20.0)
    prediction = _prediction(probability=0.9)
    transfer = TransferCost(gpu_to_ram_ms=0.0, ram_to_gpu_ms=2.0)

    benefit = model.benefit(
        prediction,
        transfer,
        eviction_cost_ms=5.0,
    )

    assert benefit == pytest.approx(11.0)


def test_bandwidth_changes_transfer_cost() -> None:
    model = CostModel()
    prediction = _prediction(size_mb=200.0)
    transfer = TransferCost(gpu_to_ram_ms=0.0, ram_to_gpu_ms=1.0)

    benefit = model.benefit(
        prediction,
        transfer,
        reuse_cost_ms=20.0,
        bandwidth_mb_s=1000.0,
    )

    assert model.transfer_cost(
        prediction,
        transfer,
        bandwidth_mb_s=1000.0,
    ) == pytest.approx(200.0)

    assert benefit == pytest.approx(
        prediction.probability * 20.0 - 200.0
    )


def test_policy_prefetches_positive_roi_with_capacity() -> None:
    model = CostModel(default_reuse_cost_ms=20.0)
    policy = SchedulerPolicy(cost_model=model)
    cache = CapacityManager(capacity_mb=512.0, used_mb=100.0)

    decision = policy.decide(
        _prediction(probability=0.9, size_mb=100.0),
        TransferCost(gpu_to_ram_ms=0.0, ram_to_gpu_ms=2.0),
        cache,
    )

    assert decision.action is SchedulerAction.PREFETCH
    assert decision.score == pytest.approx(16.0)


def test_policy_waits_when_roi_is_non_positive() -> None:
    model = CostModel(default_reuse_cost_ms=2.0)
    policy = SchedulerPolicy(cost_model=model)
    cache = CapacityManager(capacity_mb=512.0, used_mb=100.0)

    decision = policy.decide(
        _prediction(probability=0.5, size_mb=100.0),
        TransferCost(gpu_to_ram_ms=0.0, ram_to_gpu_ms=3.0),
        cache,
    )

    assert decision.action is SchedulerAction.WAIT
    assert decision.score < 0.0


def test_policy_reports_eviction_when_capacity_is_insufficient() -> None:
    model = CostModel(default_reuse_cost_ms=20.0)
    policy = SchedulerPolicy(cost_model=model)
    cache = CapacityManager(capacity_mb=128.0, used_mb=100.0)

    decision = policy.decide(
        _prediction(probability=0.9, size_mb=100.0),
        TransferCost(gpu_to_ram_ms=0.0, ram_to_gpu_ms=2.0),
        cache,
    )

    assert decision.action is SchedulerAction.EVICT
    assert decision.score == pytest.approx(16.0)


def test_policy_threshold_blocks_low_roi() -> None:
    model = CostModel(default_reuse_cost_ms=10.0)
    policy = SchedulerPolicy(
        cost_model=model,
        minimum_benefit=5.0,
    )
    cache = CapacityManager(capacity_mb=512.0, used_mb=0.0)

    decision = policy.decide(
        _prediction(probability=0.6, size_mb=100.0),
        TransferCost(gpu_to_ram_ms=0.0, ram_to_gpu_ms=2.0),
        cache,
    )

    assert decision.action is SchedulerAction.WAIT
    assert decision.score == pytest.approx(4.0)

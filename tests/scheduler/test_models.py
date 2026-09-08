import pytest

from predictive_cache.predictive_scheduler import (
    CacheState,
    ExpertPrediction,
    ExpertState,
    SchedulerAction,
    SchedulerDecision,
    SchedulerInput,
    TransferCost,
)


def test_expert_prediction():
    prediction = ExpertPrediction(
        expert_id=17,
        probability=0.92,
        expert_size_mb=240,
    )

    assert prediction.expert_id == 17
    assert prediction.probability == 0.92
    assert prediction.expert_size_mb == 240


def test_prediction_probability_validation():
    with pytest.raises(ValueError):
        ExpertPrediction(
            expert_id=17,
            probability=1.2,
            expert_size_mb=240,
        )


def test_prediction_size_validation():
    with pytest.raises(ValueError):
        ExpertPrediction(
            expert_id=17,
            probability=0.92,
            expert_size_mb=0,
        )


def test_transfer_cost():
    cost = TransferCost(
        gpu_to_ram_ms=1.5,
        ram_to_gpu_ms=2.0,
    )

    assert cost.gpu_to_ram_ms == 1.5
    assert cost.ram_to_gpu_ms == 2.0


def test_cache_state_free_memory():
    cache = CacheState(
        capacity_mb=1000,
        used_mb=720,
    )

    assert cache.free_mb == 280


def test_cache_state_cannot_overflow():
    with pytest.raises(ValueError):
        CacheState(
            capacity_mb=1000,
            used_mb=1200,
        )


def test_expert_state():
    state = ExpertState(
        expert_id=17,
        size_mb=240,
        on_ram=True,
    )

    assert state.on_ram
    assert not state.on_gpu


def test_expert_cannot_be_on_gpu_and_ram():
    with pytest.raises(ValueError):
        ExpertState(
            expert_id=17,
            size_mb=240,
            on_gpu=True,
            on_ram=True,
        )


def test_scheduler_input():
    scheduler_input = SchedulerInput(
        prediction=ExpertPrediction(
            expert_id=17,
            probability=0.92,
            expert_size_mb=240,
        ),
        transfer_cost=TransferCost(
            gpu_to_ram_ms=1.5,
            ram_to_gpu_ms=2.0,
        ),
        cache_state=CacheState(
            capacity_mb=1000,
            used_mb=720,
        ),
        expert_state=ExpertState(
            expert_id=17,
            size_mb=240,
            on_ram=True,
        ),
        reuse_cost=10.0,
        eviction_cost=1.0,
        bandwidth_mb_s=120000,
    )

    assert scheduler_input.prediction.expert_id == 17
    assert scheduler_input.cache_state.free_mb == 280


def test_scheduler_decision():
    decision = SchedulerDecision(
        expert_id=17,
        action=SchedulerAction.PREFETCH,
        score=8.31,
        reuse_value=9.2,
        transfer_cost=2.0,
        eviction_cost=0.0,
        reason="high prediction probability",
    )

    assert decision.action is SchedulerAction.PREFETCH
    assert decision.should_prefetch
    assert not decision.should_wait
    assert not decision.should_evict
import pytest

from predictive_cache.predictive_scheduler import (
    CapacityManager,
    CostModel,
    EvictionCandidate,
    ExpertPrediction,
    ExpertState,
    SchedulerAction,
    SchedulerPolicy,
    TransferCost,
)


def create_policy() -> SchedulerPolicy:
    return SchedulerPolicy(
        cost_model=CostModel(
            default_reuse_cost_ms=100.0,
        )
    )


def test_prefetch_when_benefit_is_positive_and_space_is_available():
    policy = create_policy()

    prediction = ExpertPrediction(
        expert_id=17,
        probability=0.9,
        expert_size_mb=512.0,
    )

    transfer = TransferCost(
        gpu_to_ram_ms=20.0,
        ram_to_gpu_ms=30.0,
    )

    cache = CapacityManager(
        capacity_mb=2048.0,
        used_mb=1024.0,
    )

    decision = policy.decide(
        prediction,
        transfer,
        cache,
    )

    assert decision.action is SchedulerAction.PREFETCH
    assert decision.should_prefetch
    assert decision.score == pytest.approx(60.0)


def test_wait_when_benefit_is_negative():
    policy = create_policy()

    prediction = ExpertPrediction(
        expert_id=81,
        probability=0.1,
        expert_size_mb=512.0,
    )

    transfer = TransferCost(
        gpu_to_ram_ms=20.0,
        ram_to_gpu_ms=30.0,
    )

    cache = CapacityManager(
        capacity_mb=2048.0,
        used_mb=1024.0,
    )

    decision = policy.decide(
        prediction,
        transfer,
        cache,
    )

    assert decision.action is SchedulerAction.WAIT
    assert decision.should_wait
    assert decision.score == pytest.approx(-20.0)


def test_evict_when_benefit_is_positive_but_space_is_insufficient():
    policy = create_policy()

    prediction = ExpertPrediction(
        expert_id=32,
        probability=0.9,
        expert_size_mb=1024.0,
    )

    transfer = TransferCost(
        gpu_to_ram_ms=20.0,
        ram_to_gpu_ms=30.0,
    )

    cache = CapacityManager(
        capacity_mb=2048.0,
        used_mb=1536.0,
    )

    decision = policy.decide(
        prediction,
        transfer,
        cache,
    )

    assert decision.action is SchedulerAction.EVICT
    assert decision.should_evict
    assert decision.score == pytest.approx(60.0)


def test_select_evictions():
    policy = create_policy()

    prediction = ExpertPrediction(
        expert_id=32,
        probability=0.9,
        expert_size_mb=1024.0,
    )

    cache = CapacityManager(
        capacity_mb=2048.0,
        used_mb=1536.0,
    )

    candidates = [
        # Priority 越低越优先 Evict
        __import__(
            "predictive_cache.predictive_scheduler",
            fromlist=["EvictionCandidate"],
        ).EvictionCandidate(
            expert_id=17,
            size_mb=512.0,
            eviction_priority=0.1,
        ),
        __import__(
            "predictive_cache.predictive_scheduler",
            fromlist=["EvictionCandidate"],
        ).EvictionCandidate(
            expert_id=81,
            size_mb=512.0,
            eviction_priority=0.2,
        ),
        __import__(
            "predictive_cache.predictive_scheduler",
            fromlist=["EvictionCandidate"],
        ).EvictionCandidate(
            expert_id=64,
            size_mb=512.0,
            eviction_priority=0.8,
        ),
    ]

    selected = policy.select_evictions(
        prediction,
        cache,
        candidates,
    )

    assert [item.expert_id for item in selected] == [17]

    assert sum(item.size_mb for item in selected) >= 512.0

def test_select_no_eviction_when_space_is_available():
    policy = create_policy()

    prediction = ExpertPrediction(
        expert_id=17,
        probability=0.9,
        expert_size_mb=512.0,
    )

    cache = CapacityManager(
        capacity_mb=2048.0,
        used_mb=1024.0,
    )

    candidates = []

    selected = policy.select_evictions(
        prediction,
        cache,
        candidates,
    )

    assert selected == []


def test_eviction_candidates_from_states():
    policy = create_policy()

    states = [
        ExpertState(
            expert_id=17,
            size_mb=512.0,
            on_gpu=True,
        ),
        ExpertState(
            expert_id=81,
            size_mb=768.0,
            on_gpu=True,
        ),
        ExpertState(
            expert_id=32,
            size_mb=512.0,
            on_gpu=False,
            on_ram=True,
        ),
    ]

    candidates = policy.eviction_candidates_from_states(
        states,
        priorities={
            17: 0.3,
            81: 0.8,
        },
    )

    assert [item.expert_id for item in candidates] == [
        17,
        81,
    ]
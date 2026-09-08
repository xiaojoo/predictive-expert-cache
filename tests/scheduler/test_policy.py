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


def create_policy(
    minimum_benefit: float = 0.0,
) -> SchedulerPolicy:
    return SchedulerPolicy(
        cost_model=CostModel(
            default_reuse_cost_ms=100.0,
        ),
        minimum_benefit=minimum_benefit,
    )


def test_prefetch_when_benefit_is_positive_and_space_is_available():
    policy = create_policy()

    prediction = ExpertPrediction(
        expert_id=17,
        probability=0.9,
        expert_size_mb=256.0,
    )

    transfer_cost = TransferCost(
        gpu_to_ram_ms=5.0,
        ram_to_gpu_ms=10.0,
    )

    cache = CapacityManager(
        capacity_mb=4096.0,
        used_mb=1024.0,
    )

    decision = policy.decide(
        prediction,
        transfer_cost,
        cache,
    )

    assert decision.action is SchedulerAction.PREFETCH
    assert decision.expert_id == 17
    assert decision.score > 0
    assert decision.should_prefetch is True
    assert decision.should_wait is False
    assert decision.should_evict is False


def test_wait_when_benefit_is_negative():
    policy = create_policy()

    prediction = ExpertPrediction(
        expert_id=81,
        probability=0.1,
        expert_size_mb=256.0,
    )

    transfer_cost = TransferCost(
        gpu_to_ram_ms=5.0,
        ram_to_gpu_ms=20.0,
    )

    cache = CapacityManager(
        capacity_mb=4096.0,
        used_mb=1024.0,
    )

    decision = policy.decide(
        prediction,
        transfer_cost,
        cache,
    )

    assert decision.action is SchedulerAction.WAIT
    assert decision.expert_id == 81
    assert decision.score < 0
    assert decision.should_wait is True


def test_evict_when_benefit_is_positive_but_space_is_insufficient():
    policy = create_policy()

    prediction = ExpertPrediction(
        expert_id=32,
        probability=0.95,
        expert_size_mb=2048.0,
    )

    transfer_cost = TransferCost(
        gpu_to_ram_ms=5.0,
        ram_to_gpu_ms=10.0,
    )

    cache = CapacityManager(
        capacity_mb=4096.0,
        used_mb=3072.0,
    )

    decision = policy.decide(
        prediction,
        transfer_cost,
        cache,
    )

    assert decision.action is SchedulerAction.EVICT
    assert decision.expert_id == 32
    assert decision.score > 0
    assert decision.should_evict is True


def test_select_evictions():
    policy = create_policy()

    prediction = ExpertPrediction(
        expert_id=104,
        probability=0.9,
        expert_size_mb=1024.0,
    )

    cache = CapacityManager(
        capacity_mb=2048.0,
        used_mb=1536.0,
    )

    candidates = [
        EvictionCandidate(
            expert_id=17,
            size_mb=512.0,
            eviction_priority=0.1,
        ),
        EvictionCandidate(
            expert_id=81,
            size_mb=512.0,
            eviction_priority=0.2,
        ),
        EvictionCandidate(
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


def test_select_no_eviction_when_space_is_available():
    policy = create_policy()

    prediction = ExpertPrediction(
        expert_id=17,
        probability=0.9,
        expert_size_mb=512.0,
    )

    cache = CapacityManager(
        capacity_mb=4096.0,
        used_mb=1024.0,
    )

    candidates = [
        EvictionCandidate(
            expert_id=81,
            size_mb=512.0,
            eviction_priority=0.1,
        ),
    ]

    selected = policy.select_evictions(
        prediction,
        cache,
        candidates,
    )

    assert selected == []


def test_eviction_candidates_from_states():
    states = [
        ExpertState(
            expert_id=17,
            size_mb=512.0,
            on_gpu=True,
        ),
        ExpertState(
            expert_id=81,
            size_mb=1024.0,
            on_gpu=False,
            on_ram=True,
        ),
        ExpertState(
            expert_id=32,
            size_mb=256.0,
            on_gpu=True,
        ),
    ]

    candidates = SchedulerPolicy.eviction_candidates_from_states(
        states,
        priorities={
            17: 0.2,
            32: 0.8,
        },
    )

    assert [item.expert_id for item in candidates] == [17, 32]

    assert candidates[0].size_mb == 512.0
    assert candidates[0].eviction_priority == 0.2

    assert candidates[1].size_mb == 256.0
    assert candidates[1].eviction_priority == 0.8


def test_wait_when_benefit_is_exactly_zero():
    policy = create_policy(
        minimum_benefit=0.0,
    )

    prediction = ExpertPrediction(
        expert_id=100,
        probability=0.1,
        expert_size_mb=256.0,
    )

    transfer_cost = TransferCost(
        gpu_to_ram_ms=5.0,
        ram_to_gpu_ms=10.0,
    )

    decision = policy.decide(
        prediction,
        transfer_cost,
        CapacityManager(
            capacity_mb=4096.0,
            used_mb=1024.0,
        ),
        reuse_cost_ms=100.0,
    )

    assert decision.score == pytest.approx(0.0)
    assert decision.action is SchedulerAction.WAIT
    assert decision.should_wait is True


def test_prefetch_when_cache_space_exactly_matches_expert_size():
    policy = create_policy()

    prediction = ExpertPrediction(
        expert_id=101,
        probability=0.9,
        expert_size_mb=1024.0,
    )

    transfer_cost = TransferCost(
        gpu_to_ram_ms=5.0,
        ram_to_gpu_ms=10.0,
    )

    cache = CapacityManager(
        capacity_mb=4096.0,
        used_mb=3072.0,
    )

    decision = policy.decide(
        prediction,
        transfer_cost,
        cache,
    )

    assert cache.free_mb == pytest.approx(1024.0)
    assert decision.action is SchedulerAction.PREFETCH


def test_evict_when_cache_is_short_by_one_mb():
    policy = create_policy()

    prediction = ExpertPrediction(
        expert_id=102,
        probability=0.9,
        expert_size_mb=1025.0,
    )

    transfer_cost = TransferCost(
        gpu_to_ram_ms=5.0,
        ram_to_gpu_ms=10.0,
    )

    cache = CapacityManager(
        capacity_mb=4096.0,
        used_mb=3072.0,
    )

    decision = policy.decide(
        prediction,
        transfer_cost,
        cache,
    )

    assert cache.free_mb == pytest.approx(1024.0)
    assert decision.action is SchedulerAction.EVICT


def test_select_evictions_raises_when_candidates_are_insufficient():
    policy = create_policy()

    prediction = ExpertPrediction(
        expert_id=103,
        probability=0.9,
        expert_size_mb=2048.0,
    )

    cache = CapacityManager(
        capacity_mb=4096.0,
        used_mb=3072.0,
    )

    candidates = [
        EvictionCandidate(
            expert_id=17,
            size_mb=256.0,
            eviction_priority=0.1,
        ),
        EvictionCandidate(
            expert_id=81,
            size_mb=256.0,
            eviction_priority=0.2,
        ),
    ]

    with pytest.raises(ValueError, match="insufficient eviction candidates"):
        policy.select_evictions(
            prediction,
            cache,
            candidates,
        )


def test_select_evictions_orders_by_priority_then_size():
    policy = create_policy()

    prediction = ExpertPrediction(
        expert_id=104,
        probability=0.9,
        expert_size_mb=1600.0,
    )

    cache = CapacityManager(
        capacity_mb=4096.0,
        used_mb=3072.0,
    )

    candidates = [
        EvictionCandidate(
            expert_id=1,
            size_mb=256.0,
            eviction_priority=0.5,
        ),
        EvictionCandidate(
            expert_id=2,
            size_mb=512.0,
            eviction_priority=0.5,
        ),
        EvictionCandidate(
            expert_id=3,
            size_mb=512.0,
            eviction_priority=0.1,
        ),
        EvictionCandidate(
            expert_id=4,
            size_mb=256.0,
            eviction_priority=0.8,
        ),
    ]

    selected = policy.select_evictions(
        prediction,
        cache,
        candidates,
    )

    assert [item.expert_id for item in selected] == [3, 2]


def test_eviction_candidates_from_states_ignores_ram_only_experts():
    states = [
        ExpertState(
            expert_id=17,
            size_mb=512.0,
            on_gpu=True,
        ),
        ExpertState(
            expert_id=81,
            size_mb=1024.0,
            on_ram=True,
        ),
        ExpertState(
            expert_id=32,
            size_mb=256.0,
            on_gpu=True,
        ),
    ]

    candidates = SchedulerPolicy.eviction_candidates_from_states(
        states,
        priorities={
            17: 0.1,
            81: 0.2,
            32: 0.3,
        },
    )

    assert [item.expert_id for item in candidates] == [17, 32]
    assert 81 not in [item.expert_id for item in candidates]
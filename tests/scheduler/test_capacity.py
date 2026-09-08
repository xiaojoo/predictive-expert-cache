import pytest

from predictive_cache.predictive_scheduler import (
    CapacityManager,
    EvictionCandidate,
    ExpertState,
)


def test_free_capacity():
    manager = CapacityManager(
        capacity_mb=2048.0,
        used_mb=1024.0,
    )

    assert manager.free_mb == pytest.approx(1024.0)


def test_can_fit():
    manager = CapacityManager(
        capacity_mb=2048.0,
        used_mb=1024.0,
    )

    assert manager.can_fit(512.0)
    assert manager.can_fit(1024.0)
    assert not manager.can_fit(1025.0)


def test_required_free_capacity():
    manager = CapacityManager(
        capacity_mb=2048.0,
        used_mb=1800.0,
    )

    assert manager.required_free_mb(100.0) == pytest.approx(0.0)
    assert manager.required_free_mb(512.0) == pytest.approx(264.0)


def test_reserve():
    manager = CapacityManager(
        capacity_mb=2048.0,
        used_mb=1024.0,
    )

    updated = manager.reserve(512.0)

    assert manager.used_mb == pytest.approx(1024.0)
    assert updated.used_mb == pytest.approx(1536.0)
    assert updated.free_mb == pytest.approx(512.0)


def test_reserve_insufficient_capacity():
    manager = CapacityManager(
        capacity_mb=2048.0,
        used_mb=1800.0,
    )

    with pytest.raises(ValueError):
        manager.reserve(512.0)


def test_release():
    manager = CapacityManager(
        capacity_mb=2048.0,
        used_mb=1536.0,
    )

    updated = manager.release(512.0)

    assert updated.used_mb == pytest.approx(1024.0)
    assert updated.free_mb == pytest.approx(1024.0)


def test_release_too_much():
    manager = CapacityManager(
        capacity_mb=2048.0,
        used_mb=512.0,
    )

    with pytest.raises(ValueError):
        manager.release(1024.0)


def test_select_eviction_candidates():
    candidates = [
        EvictionCandidate(
            expert_id=17,
            size_mb=512.0,
            eviction_priority=0.5,
        ),
        EvictionCandidate(
            expert_id=81,
            size_mb=768.0,
            eviction_priority=0.2,
        ),
        EvictionCandidate(
            expert_id=32,
            size_mb=256.0,
            eviction_priority=0.8,
        ),
    ]

    selected = CapacityManager.select_eviction_candidates(
        candidates,
        required_free_mb=700.0,
    )

    assert [item.expert_id for item in selected] == [81]


def test_select_multiple_eviction_candidates():
    candidates = [
        EvictionCandidate(
            expert_id=17,
            size_mb=512.0,
            eviction_priority=0.2,
        ),
        EvictionCandidate(
            expert_id=81,
            size_mb=256.0,
            eviction_priority=0.3,
        ),
        EvictionCandidate(
            expert_id=32,
            size_mb=256.0,
            eviction_priority=0.4,
        ),
    ]

    selected = CapacityManager.select_eviction_candidates(
        candidates,
        required_free_mb=700.0,
    )

    assert [item.expert_id for item in selected] == [17, 81]
    assert sum(item.size_mb for item in selected) >= 700.0


def test_candidates_from_states():
    states = [
        ExpertState(
            expert_id=17,
            size_mb=512.0,
            on_gpu=True,
        ),
        ExpertState(
            expert_id=81,
            size_mb=768.0,
            on_gpu=False,
            on_ram=True,
        ),
        ExpertState(
            expert_id=32,
            size_mb=256.0,
            on_gpu=True,
        ),
    ]

    candidates = CapacityManager.candidates_from_states(
        states,
        priorities={
            17: 0.5,
            32: 0.2,
        },
    )

    assert [item.expert_id for item in candidates] == [17, 32]
    assert candidates[0].eviction_priority == pytest.approx(0.5)
    assert candidates[1].eviction_priority == pytest.approx(0.2)


def test_no_eviction_needed():
    candidates = [
        EvictionCandidate(
            expert_id=17,
            size_mb=512.0,
            eviction_priority=0.5,
        )
    ]

    selected = CapacityManager.select_eviction_candidates(
        candidates,
        required_free_mb=0.0,
    )

    assert selected == []


def test_insufficient_eviction_candidates():
    candidates = [
        EvictionCandidate(
            expert_id=17,
            size_mb=256.0,
            eviction_priority=0.5,
        )
    ]

    with pytest.raises(ValueError):
        CapacityManager.select_eviction_candidates(
            candidates,
            required_free_mb=512.0,
        )
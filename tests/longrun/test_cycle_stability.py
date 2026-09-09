from __future__ import annotations

from dataclasses import dataclass

from predictive_cache.cache import PredictiveExpertCache
from predictive_cache.scheduler import ExpertScheduler
from predictive_cache.types import CacheConfig


@dataclass(frozen=True, slots=True)
class CycleSnapshot:
    cycle: int
    cache_size: int
    cache_capacity: int
    prediction_count: int
    request_count: int
    unique_request_count: int
    resident_count: int


def _build_trace() -> list[list[int]]:
    return [
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
        [5, 6, 7],
        [6, 7, 8],
        [7, 8, 9],
        [8, 9, 10],
        [9, 10, 11],
        [10, 11, 12],
        [11, 12, 13],
        [12, 13, 14],
        [13, 14, 15],
        [14, 15, 0],
        [15, 0, 1],
    ]


def _run_cycle(
    cache: PredictiveExpertCache,
    scheduler: ExpertScheduler,
    current_experts: list[int],
    cycle: int,
) -> CycleSnapshot:
    cache.observe(current_experts)

    predictions = cache.predict(current_experts)

    requests = scheduler.plan_predictions(
        predictions,
    )

    resident = set(cache.cached_experts())
    request_ids = [
        request.expert_id
        for request in requests
    ]

    # ---------------------------------------------------------
    # Per-cycle invariants
    # ---------------------------------------------------------

    # Cache must never exceed its configured capacity.
    assert cache.size <= cache.capacity

    # Every request must target a non-resident expert.
    assert not resident.intersection(request_ids)

    # Scheduler output must be unique.
    assert len(request_ids) == len(set(request_ids))

    # Priority must be deterministic and descending.
    priorities = [
        request.priority
        for request in requests
    ]

    assert priorities == sorted(
        priorities,
        reverse=True,
    )

    # Prediction metadata must remain valid.
    for prediction in predictions:
        assert prediction.score >= 0.0

        if prediction.estimated_distance is not None:
            assert prediction.estimated_distance >= 0.0

    return CycleSnapshot(
        cycle=cycle,
        cache_size=cache.size,
        cache_capacity=cache.capacity,
        prediction_count=len(predictions),
        request_count=len(requests),
        unique_request_count=len(set(request_ids)),
        resident_count=len(resident),
    )


def _build_runtime() -> tuple[
    PredictiveExpertCache,
    ExpertScheduler,
]:
    cache = PredictiveExpertCache(
        CacheConfig(
            capacity=8,
            prediction_top_k=4,
            recent_window=32,
        )
    )

    scheduler = ExpertScheduler(
        cache,
        cache_capacity_mb=8.0,
        default_expert_size_mb=1.0,
    )

    return cache, scheduler


def test_long_run_cycle_stability_100_cycles() -> None:
    cache, scheduler = _build_runtime()
    trace = _build_trace()

    for cycle in range(100):
        current_experts = trace[
            cycle % len(trace)
        ]

        snapshot = _run_cycle(
            cache,
            scheduler,
            current_experts,
            cycle,
        )

        assert snapshot.cache_size <= (
            snapshot.cache_capacity
        )

        assert snapshot.request_count == (
            snapshot.unique_request_count
        )

        assert snapshot.resident_count == (
            snapshot.cache_size
        )


def test_long_run_cycle_stability_1000_cycles() -> None:
    cache, scheduler = _build_runtime()
    trace = _build_trace()

    snapshots: list[CycleSnapshot] = []

    for cycle in range(1000):
        current_experts = trace[
            cycle % len(trace)
        ]

        snapshot = _run_cycle(
            cache,
            scheduler,
            current_experts,
            cycle,
        )

        snapshots.append(snapshot)

    assert len(snapshots) == 1000

    # Capacity must remain invariant for the
    # entire lifetime of the scheduler.
    capacities = {
        snapshot.cache_capacity
        for snapshot in snapshots
    }

    assert capacities == {cache.capacity}

    # Every snapshot must respect capacity.
    assert all(
        snapshot.cache_size
        <= snapshot.cache_capacity
        for snapshot in snapshots
    )

    # Scheduler must never generate duplicates.
    assert all(
        snapshot.request_count
        == snapshot.unique_request_count
        for snapshot in snapshots
    )

    # Resident count must match actual cache size.
    assert all(
        snapshot.resident_count
        == snapshot.cache_size
        for snapshot in snapshots
    )


def test_long_run_prediction_scores_remain_bounded() -> None:
    cache, scheduler = _build_runtime()
    trace = _build_trace()

    del scheduler

    for cycle in range(1000):
        current_experts = trace[
            cycle % len(trace)
        ]

        cache.observe(current_experts)

        predictions = cache.predict(
            current_experts,
        )

        for prediction in predictions:
            assert 0.0 <= prediction.score <= 1.0


def test_cycle_snapshot_is_deterministic() -> None:
    trace = _build_trace()

    cache_a, scheduler_a = _build_runtime()
    cache_b, scheduler_b = _build_runtime()

    snapshots_a = []
    snapshots_b = []

    for cycle in range(100):
        current_experts = trace[
            cycle % len(trace)
        ]

        snapshots_a.append(
            _run_cycle(
                cache_a,
                scheduler_a,
                current_experts,
                cycle,
            )
        )

        snapshots_b.append(
            _run_cycle(
                cache_b,
                scheduler_b,
                current_experts,
                cycle,
            )
        )

    assert snapshots_a == snapshots_b

def test_long_run_cache_residency_stability_1000_cycles() -> None:
    cache, scheduler = _build_runtime()
    trace = _build_trace()

    evictions = 0
    snapshots: list[tuple[int, tuple[int, ...]]] = []

    for cycle in range(1000):
        current_experts = trace[
            cycle % len(trace)
        ]

        cache.observe(current_experts)

        predictions = cache.predict(
            current_experts,
        )

        requests = scheduler.plan_predictions(
            predictions,
        )

        # Simulate successful prefetch completion.
        #
        # The real prefetch pipeline will eventually synchronize
        # residency. At this layer we only validate the cache
        # residency/LRU lifecycle.
        for request in requests:
            evicted = cache.insert(
                request.expert_id,
                size_bytes=1,
                location="memory",
            )

            if evicted is not None:
                evictions += 1

                # The evicted expert must no longer be resident.
                assert not cache.contains(evicted)

            resident = cache.cached_experts()

            # Capacity invariant.
            assert len(resident) <= cache.capacity

            # No duplicate residency.
            assert len(resident) == len(set(resident))

        snapshots.append(
            (
                cache.size,
                tuple(cache.cached_experts()),
            )
        )

    assert len(snapshots) == 1000

    # The cache must have exercised eviction under this workload.
    assert evictions > 0

    # Final state must still obey all residency invariants.
    final_resident = cache.cached_experts()

    assert cache.size == len(final_resident)
    assert cache.size <= cache.capacity
    assert len(final_resident) == len(set(final_resident))


def test_long_run_cache_residency_is_deterministic() -> None:
    trace = _build_trace()

    cache_a, scheduler_a = _build_runtime()
    cache_b, scheduler_b = _build_runtime()

    snapshots_a = []
    snapshots_b = []

    for cycle in range(1000):
        current_experts = trace[
            cycle % len(trace)
        ]

        # -----------------------------------------------------
        # Runtime A
        # -----------------------------------------------------

        cache_a.observe(current_experts)

        predictions_a = cache_a.predict(
            current_experts,
        )

        requests_a = scheduler_a.plan_predictions(
            predictions_a,
        )

        for request in requests_a:
            cache_a.insert(
                request.expert_id,
                size_bytes=1,
                location="memory",
            )

        snapshots_a.append(
            tuple(cache_a.cached_experts())
        )

        # -----------------------------------------------------
        # Runtime B
        # -----------------------------------------------------

        cache_b.observe(current_experts)

        predictions_b = cache_b.predict(
            current_experts,
        )

        requests_b = scheduler_b.plan_predictions(
            predictions_b,
        )

        for request in requests_b:
            cache_b.insert(
                request.expert_id,
                size_bytes=1,
                location="memory",
            )

        snapshots_b.append(
            tuple(cache_b.cached_experts())
        )

    assert snapshots_a == snapshots_b
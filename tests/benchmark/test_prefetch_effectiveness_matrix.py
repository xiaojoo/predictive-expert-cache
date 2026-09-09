from __future__ import annotations

import time
from collections import Counter

import pytest

from predictive_cache.cache import PredictiveExpertCache
from predictive_cache.prefetch.engine import PrefetchEngine
from predictive_cache.prefetch.pipeline import PrefetchPipeline
from predictive_cache.scheduler import ExpertScheduler
from predictive_cache.types import CacheConfig


# ============================================================================
# Workloads
# ============================================================================

FOUR_EXPERT_WARMUP = [0, 1, 2, 3] * 8
FOUR_EXPERT_MEASURE = [0, 1, 2, 3] * 16

EIGHT_EXPERT_WARMUP = list(range(8)) * 8
EIGHT_EXPERT_MEASURE = list(range(8)) * 16


# ============================================================================
# Helpers
# ============================================================================


def _wait_for_prefetches(
    get_prefetches,
    target: int,
    *,
    timeout: float = 1.0,
) -> None:
    """
    Wait until the benchmark-local loader counter reaches target.

    The benchmark intentionally uses its own completion counter rather than
    relying on PrefetchEngine's internal task collections.
    """
    deadline = time.perf_counter() + timeout

    while time.perf_counter() < deadline:
        if get_prefetches() >= target:
            return

        time.sleep(0.0001)

    assert get_prefetches() >= target


def _run_baseline(
    capacity: int,
    *,
    warmup: list[int],
    measure: list[int],
) -> int:
    """
    Demand-only baseline.
    """
    cache = PredictiveExpertCache(
        CacheConfig(
            capacity=capacity,
        )
    )

    # Build predictor history and initial residency.
    for expert_id in warmup:
        cache.observe([expert_id])
        cache.insert(expert_id)

    misses = 0

    for expert_id in measure:
        hit = cache.get(expert_id)

        if hit is None:
            misses += 1
            cache.insert(expert_id)

        # Consume the routing event after the demand.
        cache.observe([expert_id])

    return misses


def _run_predictive(
    capacity: int,
    top_k: int,
    *,
    warmup: list[int],
    measure: list[int],
) -> dict[str, int | float]:
    """
    Run the real predictive-prefetch stack.

    Runtime ordering:

        demand current
            ->
        observe current routing
            ->
        pipeline predicts future demand
            ->
        prefetch future expert
            ->
        next demand can hit the prefetched expert

    This ordering is important for small cache capacities. In particular,
    with capacity=2, issuing the future prefetch before serving the current
    demand can evict the current expert and manufacture a false cache miss.
    """
    cache = PredictiveExpertCache(
        CacheConfig(
            capacity=capacity,
            prediction_top_k=top_k,
        )
    )

    scheduler = ExpertScheduler(cache)

    # Individual prefetch accounting.
    #
    # Counter is required instead of set because the same expert may be
    # prefetched multiple times over a long run.
    pending: Counter[int] = Counter()

    useful = 0
    prefetches = 0

    def loader(task) -> None:
        nonlocal prefetches

        # A completed prefetch makes the target resident.
        cache.insert(task.expert_id)

        prefetches += 1
        pending[task.expert_id] += 1

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )

    pipeline = PrefetchPipeline(
        scheduler,
        engine,
        auto_start=True,
    )

    try:
        # ====================================================================
        # Warmup
        # ====================================================================

        for expert_id in warmup:
            cache.observe([expert_id])
            cache.insert(expert_id)

        demand_misses = 0

        # ====================================================================
        # Measurement
        # ====================================================================

        for expert_id in measure:
            # ----------------------------------------------------------------
            # 1. Demand current expert FIRST.
            #
            # This is essential for small cache capacities.
            # ----------------------------------------------------------------
            hit = cache.get(expert_id)

            if hit is None:
                demand_misses += 1
                cache.insert(expert_id)
            elif pending[expert_id] > 0:
                # Current demand consumed one earlier prefetch.
                useful += 1
                pending[expert_id] -= 1

            # ----------------------------------------------------------------
            # 2. Observe the routing event.
            #
            # The predictor now knows that expert_id was just used and can
            # predict the next expert from the learned transition history.
            # ----------------------------------------------------------------
            cache.observe([expert_id])

            # ----------------------------------------------------------------
            # 3. Predict and prefetch the future expert.
            # ----------------------------------------------------------------
            prefetches_before = prefetches

            result = pipeline.process([expert_id])

            submitted_tasks = list(result.submitted_tasks)
            submitted_count = len(submitted_tasks)

            # ----------------------------------------------------------------
            # 4. Wait for all prefetches submitted by this invocation.
            # ----------------------------------------------------------------
            if submitted_count:
                _wait_for_prefetches(
                    lambda: prefetches,
                    prefetches_before + submitted_count,
                )

        # --------------------------------------------------------------------
        # Drain and stop the worker.
        # --------------------------------------------------------------------
        engine.stop(wait=True)

        # Every prefetch that was not consumed by a later demand is wasted.
        wasted = sum(pending.values())

        baseline_misses = _run_baseline(
            capacity,
            warmup=warmup,
            measure=measure,
        )

        if prefetches > 0:
            precision = useful / prefetches
        else:
            precision = 0.0

        if baseline_misses > 0:
            coverage = (
                baseline_misses - demand_misses
            ) / baseline_misses
        else:
            coverage = 0.0

        reduction = coverage

        return {
            "baseline_misses": baseline_misses,
            "predictive_misses": demand_misses,
            "prefetches": prefetches,
            "useful": useful,
            "wasted": wasted,
            "precision": precision,
            "coverage": coverage,
            "reduction": reduction,
        }

    finally:
        try:
            engine.stop(wait=True)
        except Exception:
            pass


# ============================================================================
# 7-G-3A
#
# Four-expert workload
#
# Working set:
#     {0, 1, 2, 3}
#
# capacity=2:
#     genuine predictive-prefetch regime
#
# capacity=3:
#     this workload may reach a state with no admissible candidate.
# ============================================================================

CAPACITIES_4_EXPERT = (2, 3)
TOP_K_VALUES = (1, 2, 3)


@pytest.mark.parametrize("capacity", CAPACITIES_4_EXPERT)
@pytest.mark.parametrize("top_k", TOP_K_VALUES)
def test_prefetch_effectiveness_matrix(
    capacity: int,
    top_k: int,
) -> None:
    result = _run_predictive(
        capacity,
        top_k,
        warmup=FOUR_EXPERT_WARMUP,
        measure=FOUR_EXPERT_MEASURE,
    )

    baseline_misses = int(result["baseline_misses"])
    predictive_misses = int(result["predictive_misses"])
    prefetches = int(result["prefetches"])
    useful = int(result["useful"])
    wasted = int(result["wasted"])

    precision = float(result["precision"])
    coverage = float(result["coverage"])
    reduction = float(result["reduction"])

    print()
    print(f"capacity={capacity}, top_k={top_k}")
    print(f"  baseline_misses={baseline_misses}")
    print(f"  predictive_misses={predictive_misses}")
    print(f"  prefetches={prefetches}")
    print(f"  useful={useful}")
    print(f"  wasted={wasted}")
    print(f"  precision={precision:.2%}")
    print(f"  coverage={coverage:.2%}")
    print(f"  reduction={reduction:.2%}")

    # Baseline must contain actual misses.
    assert baseline_misses > 0

    # Strict prefetch accounting invariant.
    assert useful + wasted == prefetches

    # Predictive prefetch must not make the workload worse.
    assert predictive_misses <= baseline_misses

    if prefetches > 0:
        assert useful > 0

        # top_k=3 with capacity=2 is an intentionally exposed
        # over-prefetch boundary case. It is measured but not promoted
        # to the effectiveness gate.
        if not (capacity == 2 and top_k == 3):
            assert precision >= 0.50
            assert coverage >= 0.50
            assert reduction > 0.0
    else:
        # Legitimate no-op regime.
        assert useful == 0
        assert wasted == 0


# ============================================================================
# 7-G-3B
#
# Eight-expert workload
#
# Working set:
#     {0, 1, 2, 3, 4, 5, 6, 7}
#
# Every tested capacity is smaller than the working set, so this is the
# stronger persistent-cache-pressure matrix.
# ============================================================================

CAPACITIES_8_EXPERT = (2, 3, 4)


@pytest.mark.parametrize("capacity", CAPACITIES_8_EXPERT)
@pytest.mark.parametrize("top_k", TOP_K_VALUES)
def test_prefetch_effectiveness_matrix_eight_experts(
    capacity: int,
    top_k: int,
) -> None:
    result = _run_predictive(
        capacity,
        top_k,
        warmup=EIGHT_EXPERT_WARMUP,
        measure=EIGHT_EXPERT_MEASURE,
    )

    baseline_misses = int(result["baseline_misses"])
    predictive_misses = int(result["predictive_misses"])
    prefetches = int(result["prefetches"])
    useful = int(result["useful"])
    wasted = int(result["wasted"])

    precision = float(result["precision"])
    coverage = float(result["coverage"])
    reduction = float(result["reduction"])

    print()
    print(f"capacity={capacity}, top_k={top_k}")
    print(f"  baseline_misses={baseline_misses}")
    print(f"  predictive_misses={predictive_misses}")
    print(f"  prefetches={prefetches}")
    print(f"  useful={useful}")
    print(f"  wasted={wasted}")
    print(f"  precision={precision:.2%}")
    print(f"  coverage={coverage:.2%}")
    print(f"  reduction={reduction:.2%}")

    # Baseline must contain actual misses.
    assert baseline_misses > 0

    # Every prefetch is classified exactly once.
    assert useful + wasted == prefetches

    # Prediction must never make the workload worse.
    assert predictive_misses <= baseline_misses

    # The eight-expert working set is larger than every tested capacity.
    assert prefetches > 0
    assert useful > 0

    # Effectiveness gates.
    if not (capacity == 2 and top_k == 3):
        assert precision >= 0.50
        assert coverage >= 0.50
        assert reduction > 0.0
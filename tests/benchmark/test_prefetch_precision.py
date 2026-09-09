from __future__ import annotations

import time

from predictive_cache.cache import PredictiveExpertCache
from predictive_cache.prefetch.engine import PrefetchEngine
from predictive_cache.prefetch.pipeline import PrefetchPipeline
from predictive_cache.scheduler import ExpertScheduler
from predictive_cache.types import CacheConfig


WARMUP = [0, 1, 2, 3] * 8
MEASURE = [0, 1, 2, 3] * 16


def _make_cache() -> PredictiveExpertCache:
    return PredictiveExpertCache(
        CacheConfig(
            capacity=2,
            prediction_top_k=1,
        )
    )


def _run_baseline() -> dict[str, int]:
    cache = _make_cache()

    demand_hits = 0
    demand_misses = 0
    demand_loads = 0

    for expert_id in WARMUP + MEASURE:
        cache.observe([expert_id])

        if cache.get(expert_id) is None:
            demand_misses += 1
            demand_loads += 1
            cache.insert(expert_id)
        else:
            demand_hits += 1

    return {
        "demand_hits": demand_hits,
        "demand_misses": demand_misses,
        "demand_loads": demand_loads,
    }


def _run_predictive() -> dict[str, int]:
    cache = _make_cache()
    scheduler = ExpertScheduler(cache)

    total_prefetches = 0
    useful_prefetches = 0

    # expert_id -> number of outstanding prefetches that have
    # not yet been consumed by demand.
    prefetched_pending: set[int] = set()

    def insert_prefetched(expert_id: int) -> None:
        nonlocal total_prefetches

        evicted = cache.insert(expert_id)
        total_prefetches += 1
        prefetched_pending.add(expert_id)

        # If inserting the prefetched expert evicts another expert,
        # that evicted expert can no longer be credited as useful.
        if evicted is not None:
            prefetched_pending.discard(evicted)

    def loader(task) -> None:
        insert_prefetched(task.expert_id)

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )

    pipeline = PrefetchPipeline(
        scheduler,
        engine,
        auto_start=True,
    )

    demand_hits = 0
    demand_misses = 0
    demand_loads = 0

    try:
        for expert_id in WARMUP + MEASURE:
            cache.observe([expert_id])

            pipeline.process([expert_id])

            # Allow the prefetch worker to publish residency.
            deadline = time.perf_counter() + 0.02

            while time.perf_counter() < deadline:
                if cache.contains(expert_id):
                    break
                time.sleep(0.00005)

            if cache.get(expert_id) is None:
                demand_misses += 1
                demand_loads += 1

                evicted = cache.insert(expert_id)
                if evicted is not None:
                    prefetched_pending.discard(evicted)
            else:
                demand_hits += 1

                if expert_id in prefetched_pending:
                    useful_prefetches += 1
                    prefetched_pending.discard(expert_id)

        engine.stop(wait=True)

    finally:
        if engine.running:
            engine.stop(wait=True)

    wasted_prefetches = (
        total_prefetches
        - useful_prefetches
    )

    return {
        "demand_hits": demand_hits,
        "demand_misses": demand_misses,
        "demand_loads": demand_loads,
        "prefetches": total_prefetches,
        "useful_prefetches": useful_prefetches,
        "wasted_prefetches": wasted_prefetches,
    }


def test_prefetch_precision_and_waste_rate() -> None:
    baseline = _run_baseline()
    predictive = _run_predictive()

    total_prefetches = predictive["prefetches"]
    useful_prefetches = predictive["useful_prefetches"]
    wasted_prefetches = predictive["wasted_prefetches"]

    assert baseline["demand_misses"] > 0
    assert total_prefetches > 0

    assert (
        useful_prefetches + wasted_prefetches
        == total_prefetches
    )

    precision = (
        useful_prefetches / total_prefetches
    )

    waste_rate = (
        wasted_prefetches / total_prefetches
    )

    coverage = (
        useful_prefetches
        / baseline["demand_misses"]
    )

    print()
    print("7-G-2 Prefetch Precision / Waste Rate")
    print("--------------------------------------")
    print(
        f"baseline demand misses:  "
        f"{baseline['demand_misses']}"
    )
    print(
        f"predictive demand misses:"
        f"{predictive['demand_misses']}"
    )
    print(
        f"total prefetches:        "
        f"{total_prefetches}"
    )
    print(
        f"useful prefetches:      "
        f"{useful_prefetches}"
    )
    print(
        f"wasted prefetches:      "
        f"{wasted_prefetches}"
    )
    print(
        f"precision:              "
        f"{precision:.2%}"
    )
    print(
        f"waste rate:             "
        f"{waste_rate:.2%}"
    )
    print(
        f"coverage of baseline:   "
        f"{coverage:.2%}"
    )

    # Gate 1: the predictor must actually generate useful work.
    assert useful_prefetches > 0

    # Gate 2: at least half of executed prefetches must eventually
    # serve a demand access.
    assert precision >= 0.50

    # Gate 3: predictive prefetch must cover a meaningful fraction
    # of baseline demand misses.
    assert coverage >= 0.50

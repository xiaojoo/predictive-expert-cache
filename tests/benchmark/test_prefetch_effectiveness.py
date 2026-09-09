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

    prefetches = 0

    def loader(task) -> None:
        nonlocal prefetches

        cache.insert(task.expert_id)
        prefetches += 1

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

            # Real production path:
            #
            # Scheduler
            #   -> PrefetchPipeline
            #   -> AdmissionController
            #   -> PrefetchEngine
            #   -> loader
            #   -> cache.insert()
            pipeline.process([expert_id])

            # Give the worker a bounded opportunity to complete.
            deadline = time.perf_counter() + 0.02

            while time.perf_counter() < deadline:
                if cache.contains(expert_id):
                    break
                time.sleep(0.00005)

            if cache.get(expert_id) is None:
                demand_misses += 1
                demand_loads += 1
                cache.insert(expert_id)
            else:
                demand_hits += 1

        engine.stop(wait=True)

    finally:
        if engine.running:
            engine.stop(wait=True)

    return {
        "demand_hits": demand_hits,
        "demand_misses": demand_misses,
        "demand_loads": demand_loads,
        "prefetches": prefetches,
    }


def test_predictive_prefetch_reduces_demand_misses() -> None:
    baseline = _run_baseline()
    predictive = _run_predictive()

    reduction = (
        baseline["demand_loads"]
        - predictive["demand_loads"]
    )

    print()
    print("7-G-1 Predictive Prefetch Effectiveness")
    print("---------------------------------------")
    print(
        f"baseline demand loads:   "
        f"{baseline['demand_loads']}"
    )
    print(
        f"predictive demand loads: "
        f"{predictive['demand_loads']}"
    )
    print(
        f"baseline demand misses:  "
        f"{baseline['demand_misses']}"
    )
    print(
        f"predictive demand misses:"
        f"{predictive['demand_misses']}"
    )
    print(
        f"prefetch operations:     "
        f"{predictive['prefetches']}"
    )
    print(
        f"demand-load reduction:   "
        f"{reduction}"
    )

    assert baseline["demand_loads"] > 0
    assert predictive["prefetches"] > 0
    assert predictive["demand_loads"] < baseline["demand_loads"]

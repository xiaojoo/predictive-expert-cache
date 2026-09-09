from __future__ import annotations

import time

from predictive_cache.cache import PredictiveExpertCache
from predictive_cache.prefetch.engine import PrefetchEngine
from predictive_cache.prefetch.pipeline import PrefetchPipeline
from predictive_cache.types import CacheConfig
from predictive_cache.scheduler import ExpertScheduler


def _wait_for_tasks(
    engine: PrefetchEngine,
    tasks,
    *,
    timeout_s: float = 2.0,
) -> None:
    deadline = time.perf_counter() + timeout_s
    expert_ids = {task.expert_id for task in tasks}

    while time.perf_counter() < deadline:
        if all(
            engine.task_status(expert_id) == "completed"
            for expert_id in expert_ids
        ):
            return

        time.sleep(0.0001)

    states = {
        expert_id: engine.task_status(expert_id)
        for expert_id in sorted(expert_ids)
    }

    raise AssertionError(
        f"prefetch tasks did not complete within benchmark timeout: {states}"
    )


def test_capacity_pressure_limits_prefetches_per_process() -> None:
    cache = PredictiveExpertCache(
        CacheConfig(
            capacity=2,
            prediction_top_k=3,
        )
    )

    scheduler = ExpertScheduler(
        cache,
        cache_capacity_mb=2.0,
        cache_used_mb=1.0,
        default_expert_size_mb=1.0,
    )

    def loader(task) -> None:
        cache.insert(task.expert_id)

    engine = PrefetchEngine(loader)

    pipeline = PrefetchPipeline(
        scheduler,
        engine,
        auto_start=True,
    )

    try:
        for expert_id in [0, 1, 2, 3] * 8:
            cache.get(expert_id)
            cache.insert(expert_id)
            cache.observe([expert_id])

        admitted = []

        current_expert = 0

        cache.get(current_expert)
        if not cache.contains(current_expert):
            cache.insert(current_expert)

        cache.observe([current_expert])

        result = pipeline.process([current_expert])
        admitted.extend(result.submitted_tasks)

        if admitted:
            _wait_for_tasks(engine, admitted)

        print()
        print("7-H-3 Capacity-Aware Admission")
        print("--------------------------------")
        print(f"cache capacity:       {cache.capacity}")
        print("prediction top_k:     3")
        print(f"scheduled requests:   {len(result.scheduled_requests)}")
        print(f"submitted prefetches: {len(result.submitted_tasks)}")
        print(
            "admission decisions:  ",
            [
                (d.admitted, d.reason.value)
                for d in result.admission_decisions
            ],
        )

        assert len(result.scheduled_requests) >= 1
        assert len(result.submitted_tasks) <= 1

    finally:
        engine.stop()

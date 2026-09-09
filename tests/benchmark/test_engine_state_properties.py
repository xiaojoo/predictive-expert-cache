from __future__ import annotations

import time
from statistics import mean, median

from predictive_cache.prefetch.engine import PrefetchEngine
from predictive_cache.prefetch.types import (
    PrefetchSource,
    PrefetchTarget,
    PrefetchTask,
)


def _percentile(values: list[int], p: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p
    lo = int(rank)
    hi = min(lo + 1, len(ordered))
    weight = rank - lo
    return ordered[lo] + (
        ordered[hi] - ordered[lo]
    ) * weight


def _report(name: str, samples: list[int]) -> None:
    values = [x / 1_000.0 for x in samples]

    print()
    print(name)
    print("-" * len(name))
    print(f"cycles:     {len(values)}")
    print(f"mean:       {mean(values):.3f} us")
    print(f"median:     {median(values):.3f} us")
    print(f"p95:        {_percentile(values, 0.95):.3f} us")
    print(f"min:        {min(values):.3f} us")
    print(f"max:        {max(values):.3f} us")


def _make_task(expert_id: int) -> PrefetchTask:
    return PrefetchTask(
        expert_id=expert_id,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
        priority=1.0,
        confidence=1.0,
        estimated_distance=1,
    )


def test_7f12_engine_state_properties() -> None:
    cycles = 10_000

    def loader(task: PrefetchTask) -> None:
        pass

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )
    engine.start()

    try:
        queued_samples: list[int] = []
        running_samples: list[int] = []
        queue_full_samples: list[int] = []

        for _ in range(cycles):
            started = time.perf_counter_ns()
            value = engine.queued_tasks
            queued_samples.append(
                time.perf_counter_ns() - started
            )
            assert value is not None

        for _ in range(cycles):
            started = time.perf_counter_ns()
            value = engine.running_tasks
            running_samples.append(
                time.perf_counter_ns() - started
            )
            assert value is not None

        for _ in range(cycles):
            started = time.perf_counter_ns()
            value = engine.queue_full
            queue_full_samples.append(
                time.perf_counter_ns() - started
            )
            assert isinstance(value, bool)

        _report(
            "7-F-12 engine.queued_tasks",
            queued_samples,
        )

        _report(
            "7-F-12 engine.running_tasks",
            running_samples,
        )

        _report(
            "7-F-12 engine.queue_full",
            queue_full_samples,
        )

    finally:
        engine.stop()


if __name__ == "__main__":
    test_7f12_engine_state_properties()
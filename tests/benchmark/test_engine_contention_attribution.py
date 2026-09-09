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
    return ordered[lo] + (ordered[hi] - ordered[lo]) * weight


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


def _make_engine(loader_delay: float = 0.0) -> PrefetchEngine:
    def loader(task: PrefetchTask) -> None:
        if loader_delay:
            time.sleep(loader_delay)

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )
    engine.start()
    return engine


def _wait_completed(
    engine: PrefetchEngine,
    expert_id: int,
) -> None:
    deadline = time.monotonic() + 5.0

    while time.monotonic() < deadline:
        status = engine.task_status(expert_id)

        if status is not None and status.name == "COMPLETED":
            return

        if status is not None and status.name in (
            "FAILED",
            "CANCELLED",
        ):
            raise AssertionError(f"task failed: {status}")

        time.sleep(0.00001)

    raise AssertionError(
        f"task {expert_id} did not complete"
    )


def test_7f10_engine_contention_attribution() -> None:
    cycles = 2000

    # ---------------------------------------------------------
    # A. Real worker, extremely cheap loader.
    # ---------------------------------------------------------
    engine = _make_engine()

    try:
        submit_samples: list[int] = []
        status_samples: list[int] = []
        submit_status_samples: list[int] = []

        for i in range(cycles):
            expert_id = i % 64
            task = _make_task(expert_id)

            started = time.perf_counter_ns()
            assert engine.submit(task)
            submit_samples.append(
                time.perf_counter_ns() - started
            )

            _wait_completed(engine, expert_id)

        for i in range(cycles):
            expert_id = i % 64

            # Ensure the task has completed before measuring
            # task-status lookup in isolation.
            _wait_completed(engine, expert_id)

            started = time.perf_counter_ns()
            status = engine.task_status(expert_id)
            status_samples.append(
                time.perf_counter_ns() - started
            )

            assert status is not None

        for i in range(cycles):
            expert_id = i % 64
            task = _make_task(expert_id)

            started = time.perf_counter_ns()

            assert engine.submit(task)

            status = None
            while status is None:
                status = engine.task_status(expert_id)

            submit_status_samples.append(
                time.perf_counter_ns() - started
            )

            _wait_completed(engine, expert_id)

        _report(
            "7-F-10 Engine.submit() isolated",
            submit_samples,
        )

        _report(
            "7-F-10 Engine.task_status() isolated",
            status_samples,
        )

        _report(
            "7-F-10 submit() + first task_status()",
            submit_status_samples,
        )

    finally:
        engine.stop()


def test_7f10_engine_worker_delay_comparison() -> None:
    cycles = 1000

    results: list[tuple[str, list[int]]] = []

    for label, delay in (
        ("zero-delay worker", 0.0),
        ("100-us worker", 0.0001),
    ):
        engine = _make_engine(delay)

        try:
            samples: list[int] = []

            for i in range(cycles):
                expert_id = i % 64
                task = _make_task(expert_id)

                started = time.perf_counter_ns()
                assert engine.submit(task)
                samples.append(
                    time.perf_counter_ns() - started
                )

                _wait_completed(engine, expert_id)

            results.append((label, samples))

        finally:
            engine.stop()

    for label, samples in results:
        _report(
            f"7-F-10 submit under {label}",
            samples,
        )


if __name__ == "__main__":
    test_7f10_engine_contention_attribution()
    test_7f10_engine_worker_delay_comparison()
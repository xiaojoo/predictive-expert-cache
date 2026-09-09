from __future__ import annotations

import time
from statistics import mean, median

from predictive_cache.prefetch.admission import AdmissionController
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


def _make_engine() -> PrefetchEngine:
    def loader(task: PrefetchTask) -> None:
        pass

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


def test_7f13_admission_path_attribution() -> None:
    cycles = 10_000

    admission = AdmissionController()
    engine = _make_engine()

    try:
        priority_samples: list[int] = []
        confidence_samples: list[int] = []
        queued_membership_samples: list[int] = []
        running_membership_samples: list[int] = []
        queue_full_samples: list[int] = []
        full_admission_samples: list[int] = []

        for i in range(cycles):
            task = _make_task(i % 64)

            started = time.perf_counter_ns()
            value = task.priority < admission.min_priority
            priority_samples.append(
                time.perf_counter_ns() - started
            )
            assert value is False

        for i in range(cycles):
            task = _make_task(i % 64)

            started = time.perf_counter_ns()
            value = task.confidence < admission.min_confidence
            confidence_samples.append(
                time.perf_counter_ns() - started
            )
            assert value is False

        for i in range(cycles):
            expert_id = i % 64

            started = time.perf_counter_ns()
            value = expert_id in engine.queued_tasks
            queued_membership_samples.append(
                time.perf_counter_ns() - started
            )
            assert value is False

        for i in range(cycles):
            expert_id = i % 64

            started = time.perf_counter_ns()
            value = expert_id in engine.running_tasks
            running_membership_samples.append(
                time.perf_counter_ns() - started
            )
            assert value is False

        for _ in range(cycles):
            started = time.perf_counter_ns()
            value = engine.queue_full
            queue_full_samples.append(
                time.perf_counter_ns() - started
            )
            assert value is False

        for i in range(cycles):
            task = _make_task(i % 64)

            started = time.perf_counter_ns()

            decision = admission.evaluate(
                task,
                engine,
            )

            full_admission_samples.append(
                time.perf_counter_ns() - started
            )

            assert decision.admitted

        _report(
            "7-F-13 priority check",
            priority_samples,
        )

        _report(
            "7-F-13 confidence check",
            confidence_samples,
        )

        _report(
            "7-F-13 queued membership",
            queued_membership_samples,
        )

        _report(
            "7-F-13 running membership",
            running_membership_samples,
        )

        _report(
            "7-F-13 queue_full",
            queue_full_samples,
        )

        _report(
            "7-F-13 full Admission.evaluate()",
            full_admission_samples,
        )

    finally:
        engine.stop()


if __name__ == "__main__":
    test_7f13_admission_path_attribution()
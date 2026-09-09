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

    return (
        ordered[lo]
        + (ordered[hi] - ordered[lo]) * weight
    )


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


class _NullEngine:
    @property
    def queued_tasks(self) -> set[int]:
        return set()

    @property
    def running_tasks(self) -> set[int]:
        return set()

    @property
    def queue_full(self) -> bool:
        return False


def _make_real_engine() -> PrefetchEngine:
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
            raise AssertionError(
                f"task failed: {status}"
            )

        time.sleep(0.00001)

    raise AssertionError(
        f"task {expert_id} did not complete"
    )


def test_7f11_admission_engine_boundary() -> None:
    cycles = 2000

    admission = AdmissionController()

    null_engine = _NullEngine()
    real_engine = _make_real_engine()

    try:
        null_samples: list[int] = []
        real_samples: list[int] = []

        # -----------------------------------------------------
        # NullEngine admission.
        # -----------------------------------------------------
        for i in range(cycles):
            expert_id = i % 64
            task = _make_task(expert_id)

            started = time.perf_counter_ns()

            decision = admission.evaluate(
                task,
                null_engine,  # type: ignore[arg-type]
            )

            null_samples.append(
                time.perf_counter_ns() - started
            )

            assert decision.admitted

        # -----------------------------------------------------
        # RealEngine admission with no active task.
        # -----------------------------------------------------
        for i in range(cycles):
            expert_id = i % 64

            # Make sure the previous task is completely finished.
            if i > 0:
                _wait_completed(
                    real_engine,
                    (i - 1) % 64,
                )

            task = _make_task(expert_id)

            started = time.perf_counter_ns()

            decision = admission.evaluate(
                task,
                real_engine,
            )

            real_samples.append(
                time.perf_counter_ns() - started
            )

            assert decision.admitted

            assert real_engine.submit(task)

        # Ensure final task is complete.
        _wait_completed(
            real_engine,
            (cycles - 1) % 64,
        )

        _report(
            "7-F-11 Admission.evaluate() with NullEngine",
            null_samples,
        )

        _report(
            "7-F-11 Admission.evaluate() with RealEngine",
            real_samples,
        )

    finally:
        real_engine.stop()


if __name__ == "__main__":
    test_7f11_admission_engine_boundary()
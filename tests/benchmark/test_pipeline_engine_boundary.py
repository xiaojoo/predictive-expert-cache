from __future__ import annotations

import time
from statistics import mean, median

from predictive_cache.cache import PredictiveExpertCache
from predictive_cache.prefetch.admission import AdmissionController
from predictive_cache.prefetch.engine import PrefetchEngine
from predictive_cache.prefetch.pipeline import PrefetchPipeline
from predictive_cache.prefetch.types import (
    PrefetchSource,
    PrefetchTarget,
    PrefetchTask,
)
from predictive_cache.scheduler import ExpertScheduler
from predictive_cache.storage.expert_record import ExpertRecord
from predictive_cache.storage.expert_store import ExpertLocation
from predictive_cache.storage.in_memory import InMemoryExpertStore
from predictive_cache.prefetch.storage import (
    create_storage_transfer_executor,
)


def _percentile(values: list[int], p: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p
    lo = int(rank)
    hi = min(lo + 1, len(ordered))
    weight = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * weight


def _report(name: str, samples: list[int]) -> None:
    samples_us = [x / 1_000.0 for x in samples]

    print()
    print(name)
    print("-" * len(name))
    print(f"cycles:     {len(samples)}")
    print(f"mean:       {mean(samples_us):.3f} us")
    print(f"median:     {median(samples_us):.3f} us")
    print(f"p95:        {_percentile(samples_us, 0.95):.3f} us")
    print(f"min:        {min(samples_us):.3f} us")
    print(f"max:        {max(samples_us):.3f} us")


def _make_scheduler() -> ExpertScheduler:
    cache = PredictiveExpertCache()

    for expert_id in range(64):
        cache.observe([expert_id])

    return ExpertScheduler(
        cache,
        minimum_benefit=0.0,
    )


def _make_engine() -> PrefetchEngine:
    nvme = InMemoryExpertStore()
    ram = InMemoryExpertStore()

    for expert_id in range(64):
        nvme.put(
            ExpertRecord(
                expert_id=expert_id,
                location=ExpertLocation.NVME,
                payload=f"expert-{expert_id}",
            )
        )

    def fallback(task: PrefetchTask) -> None:
        pass

    transfer = create_storage_transfer_executor(
        nvme,
        ram,
        fallback,
    )

    engine = PrefetchEngine(
        fallback,
        num_workers=1,
        transfer_executor=transfer,
    )
    engine.start()
    return engine


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
    def __init__(self) -> None:
        self._queued: set[int] = set()
        self._running: set[int] = set()

    @property
    def queued_tasks(self):
        return self._queued

    @property
    def running_tasks(self):
        return self._running

    @property
    def queue_full(self) -> bool:
        return False

    def submit(self, task: PrefetchTask) -> bool:
        return True


def test_7f9_pipeline_engine_boundary() -> None:
    cycles = 2000

    scheduler = _make_scheduler()
    real_engine = _make_engine()

    real_pipeline = PrefetchPipeline(
        scheduler,
        real_engine,
        auto_start=False,
    )

    null_engine = _NullEngine()

    null_pipeline = PrefetchPipeline(
        scheduler,
        null_engine,  # type: ignore[arg-type]
        auto_start=False,
    )

    try:
        real_samples: list[int] = []
        null_samples: list[int] = []

        for i in range(cycles):
            expert_id = i % 64

            started = time.perf_counter_ns()
            result = null_pipeline.process([expert_id])
            null_samples.append(
                time.perf_counter_ns() - started
            )

            assert result.scheduled_requests
            assert result.submitted_tasks

        for i in range(cycles):
            expert_id = i % 64

            started = time.perf_counter_ns()
            result = real_pipeline.process([expert_id])
            real_samples.append(
                time.perf_counter_ns() - started
            )

            assert result.scheduled_requests

            for submitted in result.submitted_tasks:
                deadline = time.monotonic() + 5.0

                while time.monotonic() < deadline:
                    status = real_engine.task_status(
                        submitted.expert_id
                    )

                    if status is not None and status.name == "COMPLETED":
                        break

                    if status is not None and status.name in (
                        "FAILED",
                        "CANCELLED",
                    ):
                        raise AssertionError(
                            f"task failed: {status}"
                        )

                    time.sleep(0.00001)
                else:
                    raise AssertionError(
                        f"task {submitted.expert_id} "
                        "did not complete"
                    )

        _report(
            "7-F-9 Pipeline.process() with NullEngine",
            null_samples,
        )

        _report(
            "7-F-9 Pipeline.process() with real Engine",
            real_samples,
        )

    finally:
        real_pipeline.stop()


if __name__ == "__main__":
    test_7f9_pipeline_engine_boundary()
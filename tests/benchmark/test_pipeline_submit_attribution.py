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
    hi = min(lo + 1, len(ordered) - 1)
    weight = rank - lo

    return (
        ordered[lo]
        + (ordered[hi] - ordered[lo]) * weight
    )


def _report(name: str, samples: list[int]) -> None:
    samples_us = [
        value / 1_000.0
        for value in samples
    ]

    print()
    print(name)
    print("-" * len(name))
    print(f"cycles:     {len(samples)}")
    print(f"mean:       {mean(samples_us):.3f} us")
    print(f"median:     {median(samples_us):.3f} us")
    print(
        f"p95:        "
        f"{_percentile(samples_us, 0.95):.3f} us"
    )
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


def test_7f8_engine_submit_attribution() -> None:
    cycles = 2_000

    scheduler = _make_scheduler()
    engine = _make_engine()

    pipeline = PrefetchPipeline(
        scheduler,
        engine,
        auto_start=False,
    )

    admission = AdmissionController()

    try:
        current = [0]

        requests = scheduler.plan_prefetch_predictions(
            current
        )

        assert requests

        tasks = [
            pipeline._request_to_task(request)
            for request in requests
        ]

        print()
        print("7-F-8 Engine submit attribution")
        print("================================")
        print(f"scheduled requests: {len(tasks)}")

        # ---------------------------------------------------------
        # Single submit
        # ---------------------------------------------------------

        single_submit_samples: list[int] = []

        for i in range(cycles):
            expert_id = i % 64
            task = _make_task(expert_id)

            status = engine.task_status(expert_id)

            while status is not None and status.name not in (
                "COMPLETED",
                "FAILED",
                "CANCELLED",
            ):
                time.sleep(0.00001)
                status = engine.task_status(expert_id)

            started = time.perf_counter_ns()

            assert engine.submit(task)

            single_submit_samples.append(
                time.perf_counter_ns() - started
            )

            _wait_completed(
                engine,
                expert_id,
            )

        # ---------------------------------------------------------
        # Four sequential submits
        # ---------------------------------------------------------

        batch_submit_samples: list[int] = []

        for cycle in range(cycles):
            batch_tasks = [
                _make_task(
                    (cycle * len(tasks) + index) % 64
                )
                for index in range(len(tasks))
            ]

            for task in batch_tasks:
                status = engine.task_status(
                    task.expert_id
                )

                while status is not None and status.name not in (
                    "COMPLETED",
                    "FAILED",
                    "CANCELLED",
                ):
                    time.sleep(0.00001)
                    status = engine.task_status(
                        task.expert_id
                    )

            started = time.perf_counter_ns()

            for task in batch_tasks:
                assert engine.submit(task)

            batch_submit_samples.append(
                time.perf_counter_ns() - started
            )

            for task in batch_tasks:
                _wait_completed(
                    engine,
                    task.expert_id,
                )

        # ---------------------------------------------------------
        # Pipeline full process
        # ---------------------------------------------------------

        process_samples: list[int] = []

        for i in range(cycles):
            expert_id = i % 64

            # Ensure the previous task for this expert is finished.
            status = engine.task_status(expert_id)

            while status is not None and status.name not in (
                "COMPLETED",
                "FAILED",
                "CANCELLED",
            ):
                time.sleep(0.00001)
                status = engine.task_status(expert_id)

            started = time.perf_counter_ns()

            result = pipeline.process(
                [expert_id]
            )

            process_samples.append(
                time.perf_counter_ns() - started
            )

            assert result.scheduled_requests

            for submitted in result.submitted_tasks:
                _wait_completed(
                    engine,
                    submitted.expert_id,
                )

        _report(
            "7-F-8 Engine.submit() single task",
            single_submit_samples,
        )

        _report(
            "7-F-8 Engine.submit() batch",
            batch_submit_samples,
        )

        _report(
            "7-F-8 Pipeline.process() real submit path",
            process_samples,
        )

    finally:
        pipeline.stop()


if __name__ == "__main__":
    test_7f8_engine_submit_attribution()
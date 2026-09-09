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


def test_7f_pipeline_bottleneck_attribution() -> None:
    cycles = 10_000

    scheduler = _make_scheduler()
    engine = _make_engine()

    pipeline = PrefetchPipeline(
        scheduler,
        engine,
        auto_start=False,
    )

    requests = scheduler.plan_prefetch_predictions([0])
    assert requests

    request = requests[0]
    task = pipeline._request_to_task(request)

    admission = AdmissionController()

    try:
        scheduler_samples: list[int] = []
        task_samples: list[int] = []
        admission_samples: list[int] = []
        submit_samples: list[int] = []
        process_samples: list[int] = []

        for i in range(cycles):
            expert_id = i % 64
            current = [expert_id]

            started = time.perf_counter_ns()
            planned = scheduler.plan_prefetch_predictions(current)
            scheduler_samples.append(
                time.perf_counter_ns() - started
            )
            assert planned

        for _ in range(cycles):
            started = time.perf_counter_ns()
            converted = pipeline._request_to_task(request)
            task_samples.append(
                time.perf_counter_ns() - started
            )
            assert converted.expert_id == request.expert_id

        for _ in range(cycles):
            started = time.perf_counter_ns()
            decision = admission.evaluate(task, engine)
            admission_samples.append(
                time.perf_counter_ns() - started
            )
            assert decision.admitted

        for i in range(cycles):
            expert_id = i % 64
            submit_task = _make_task(expert_id)

            while engine.task_status(expert_id) not in (
                None,
            ):
                status = engine.task_status(expert_id)
                if status.name in (
                    "COMPLETED",
                    "FAILED",
                    "CANCELLED",
                ):
                    break
                time.sleep(0.00001)

            started = time.perf_counter_ns()
            assert engine.submit(submit_task)
            submit_samples.append(
                time.perf_counter_ns() - started
            )

            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                status = engine.task_status(expert_id)
                if status.name == "COMPLETED":
                    break
                if status.name in ("FAILED", "CANCELLED"):
                    raise AssertionError(
                        f"task failed: {status}"
                    )
                time.sleep(0.00001)
            else:
                raise AssertionError(
                    f"task {expert_id} did not complete"
                )

        for i in range(cycles):
            expert_id = i % 64

            started = time.perf_counter_ns()
            result = pipeline.process([expert_id])
            process_samples.append(
                time.perf_counter_ns() - started
            )

            assert result.scheduled_requests

            for submitted in result.submitted_tasks:
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    status = engine.task_status(
                        submitted.expert_id
                    )
                    if status.name == "COMPLETED":
                        break
                    if status.name in (
                        "FAILED",
                        "CANCELLED",
                    ):
                        raise AssertionError(
                            f"pipeline task failed: {status}"
                        )
                    time.sleep(0.00001)
                else:
                    raise AssertionError(
                        f"pipeline task {submitted.expert_id} "
                        "did not complete"
                    )

        _report(
            "7-F Scheduler.plan_prefetch_predictions()",
            scheduler_samples,
        )
        _report(
            "7-F Pipeline._request_to_task()",
            task_samples,
        )
        _report(
            "7-F AdmissionController.evaluate()",
            admission_samples,
        )
        _report(
            "7-F PrefetchEngine.submit()",
            submit_samples,
        )
        _report(
            "7-F Pipeline.process()",
            process_samples,
        )

    finally:
        pipeline.stop()


if __name__ == "__main__":
    test_7f_pipeline_bottleneck_attribution()

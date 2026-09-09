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


def test_7f14_pipeline_live_stage_attribution() -> None:
    cycles = 2000

    scheduler = _make_scheduler()
    engine = _make_engine()

    pipeline = PrefetchPipeline(
        scheduler,
        engine,
        auto_start=False,
    )

    admission = AdmissionController()

    try:
        scheduler_samples: list[int] = []
        process_samples: list[int] = []

        conversion_samples: list[int] = []
        admission_samples: list[int] = []
        submit_samples: list[int] = []

        submit_count_samples: list[int] = []

        for i in range(cycles):
            expert_id = i % 64

            # -------------------------------------------------
            # Measure scheduler separately for this same cycle.
            # -------------------------------------------------
            started = time.perf_counter_ns()

            requests = scheduler.plan_prefetch_predictions(
                [expert_id]
            )

            scheduler_samples.append(
                time.perf_counter_ns() - started
            )

            assert requests
            assert len(requests) == 4

            # -------------------------------------------------
            # Reproduce Pipeline.process() exactly, while
            # measuring every stage inside the same invocation.
            # -------------------------------------------------
            started_process = time.perf_counter_ns()

            submitted_count = 0

            for request in requests:
                started = time.perf_counter_ns()

                task = pipeline._request_to_task(request)

                conversion_samples.append(
                    time.perf_counter_ns() - started
                )

                started = time.perf_counter_ns()

                decision = admission.evaluate(
                    task,
                    engine,
                )

                admission_samples.append(
                    time.perf_counter_ns() - started
                )

                if not decision.admitted:
                    continue

                started = time.perf_counter_ns()

                accepted = engine.submit(task)

                submit_samples.append(
                    time.perf_counter_ns() - started
                )

                assert accepted

                submitted_count += 1

            process_samples.append(
                time.perf_counter_ns() - started_process
            )

            submit_count_samples.append(
                submitted_count
            )

            assert submitted_count == 4

            # Do not allow unfinished work from one iteration
            # to contaminate the next iteration.
            for offset in range(4):
                submitted_expert = requests[offset].expert_id
                _wait_completed(
                    engine,
                    submitted_expert,
                )

        _report(
            "7-F-14 scheduler",
            scheduler_samples,
        )

        _report(
            "7-F-14 request_to_task per request",
            conversion_samples,
        )

        _report(
            "7-F-14 admission per request",
            admission_samples,
        )

        _report(
            "7-F-14 submit per request",
            submit_samples,
        )

        _report(
            "7-F-14 live process body",
            process_samples,
        )

        _report(
            "7-F-14 full scheduler + process body",
            [
                scheduler_samples[i]
                + process_samples[i]
                for i in range(cycles)
            ],
        )

        assert all(
            count == 4
            for count in submit_count_samples
        )

    finally:
        pipeline.stop()


if __name__ == "__main__":
    test_7f14_pipeline_live_stage_attribution()
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


def test_7f8_pipeline_process_internal_attribution() -> None:
    cycles = 10_000

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
        print("7-F-8 Pipeline internal attribution")
        print("===================================")
        print(
            f"scheduled requests: "
            f"{len(requests)}"
        )

        # ---------------------------------------------------------
        # 1. Planner result materialization
        # ---------------------------------------------------------

        materialize_samples: list[int] = []

        for _ in range(cycles):
            started = time.perf_counter_ns()

            materialized = list(
                scheduler.plan_prefetch_predictions(
                    current
                )
            )

            materialize_samples.append(
                time.perf_counter_ns() - started
            )

            assert materialized

        # ---------------------------------------------------------
        # 2. All request -> task conversions
        # ---------------------------------------------------------

        task_conversion_samples: list[int] = []

        for _ in range(cycles):
            started = time.perf_counter_ns()

            converted = [
                pipeline._request_to_task(
                    request
                )
                for request in requests
            ]

            task_conversion_samples.append(
                time.perf_counter_ns() - started
            )

            assert converted

        # ---------------------------------------------------------
        # 3. Admission evaluation
        # ---------------------------------------------------------

        admission_samples: list[int] = []

        for _ in range(cycles):
            started = time.perf_counter_ns()

            decisions = [
                admission.evaluate(
                    task,
                    engine,
                )
                for task in tasks
            ]

            admission_samples.append(
                time.perf_counter_ns() - started
            )

            assert all(
                decision.admitted
                for decision in decisions
            )

        # ---------------------------------------------------------
        # 4. Admission + stats
        # ---------------------------------------------------------

        admission_stats_samples: list[int] = []

        for _ in range(cycles):
            started = time.perf_counter_ns()

            for task in tasks:
                decision = admission.evaluate(
                    task,
                    engine,
                )
                pipeline.admission_stats.record(
                    decision
                )

            admission_stats_samples.append(
                time.perf_counter_ns() - started
            )

        # ---------------------------------------------------------
        # 5. list / append bookkeeping
        # ---------------------------------------------------------

        append_samples: list[int] = []

        for _ in range(cycles):
            started = time.perf_counter_ns()

            submitted_tasks = []

            for task in tasks:
                submitted_tasks.append(task)

            append_samples.append(
                time.perf_counter_ns() - started
            )

            assert submitted_tasks

        # ---------------------------------------------------------
        # 6. Full synchronous planning/bookkeeping path
        #
        # No engine.submit() here.
        # This isolates Pipeline.process() CPU work from
        # worker scheduling and transfer completion.
        # ---------------------------------------------------------

        synchronous_process_samples: list[int] = []

        for _ in range(cycles):
            started = time.perf_counter_ns()

            scheduled_requests = list(
                scheduler.plan_prefetch_predictions(
                    current
                )
            )

            submitted_tasks = []

            for request in scheduled_requests:
                task = pipeline._request_to_task(
                    request
                )

                decision = admission.evaluate(
                    task,
                    engine,
                )

                pipeline.admission_stats.record(
                    decision
                )

                if not decision.admitted:
                    continue

                submitted_tasks.append(task)

            synchronous_process_samples.append(
                time.perf_counter_ns() - started
            )

            assert scheduled_requests
            assert submitted_tasks

        # ---------------------------------------------------------
        # Reports
        # ---------------------------------------------------------

        _report(
            "7-F-8 list(planner_result)",
            materialize_samples,
        )

        _report(
            "7-F-8 All _request_to_task() conversions",
            task_conversion_samples,
        )

        _report(
            "7-F-8 All AdmissionController.evaluate()",
            admission_samples,
        )

        _report(
            "7-F-8 Admission + AdmissionStats.record()",
            admission_stats_samples,
        )

        _report(
            "7-F-8 submitted_tasks.append()",
            append_samples,
        )

        _report(
            "7-F-8 Synchronous Pipeline.process() path",
            synchronous_process_samples,
        )

    finally:
        pipeline.stop()


if __name__ == "__main__":
    test_7f8_pipeline_process_internal_attribution()
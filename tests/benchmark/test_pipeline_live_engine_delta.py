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


class _NullEngine:
    def __init__(self) -> None:
        self._queued: set[int] = set()
        self._running: set[int] = set()

    @property
    def queued_tasks(self) -> set[int]:
        return self._queued

    @property
    def running_tasks(self) -> set[int]:
        return self._running

    @property
    def queue_full(self) -> bool:
        return False

    def submit(self, task: PrefetchTask) -> bool:
        return True


def _make_task_from_request(
    pipeline: PrefetchPipeline,
    request,
) -> PrefetchTask:
    return pipeline._request_to_task(request)


def test_7f15_pipeline_live_engine_delta() -> None:
    cycles = 2000

    scheduler = _make_scheduler()
    real_engine = _make_engine()
    null_engine = _NullEngine()

    real_pipeline = PrefetchPipeline(
        scheduler,
        real_engine,
        auto_start=False,
    )

    null_pipeline = PrefetchPipeline(
        scheduler,
        null_engine,  # type: ignore[arg-type]
        auto_start=False,
    )

    admission = AdmissionController()

    real_body_samples: list[int] = []
    null_body_samples: list[int] = []

    real_submit_samples: list[int] = []
    null_submit_samples: list[int] = []

    real_admission_samples: list[int] = []
    null_admission_samples: list[int] = []

    try:
        # -----------------------------------------------------
        # NULL ENGINE
        # -----------------------------------------------------
        for i in range(cycles):
            expert_id = i % 64

            requests = scheduler.plan_prefetch_predictions(
                [expert_id]
            )

            assert len(requests) == 4

            started_body = time.perf_counter_ns()

            submitted = 0

            for request in requests:
                task = _make_task_from_request(
                    null_pipeline,
                    request,
                )

                started = time.perf_counter_ns()

                decision = admission.evaluate(
                    task,
                    null_engine,  # type: ignore[arg-type]
                )

                null_admission_samples.append(
                    time.perf_counter_ns() - started
                )

                assert decision.admitted

                started = time.perf_counter_ns()

                assert null_engine.submit(task)

                null_submit_samples.append(
                    time.perf_counter_ns() - started
                )

                submitted += 1

            null_body_samples.append(
                time.perf_counter_ns() - started_body
            )

            assert submitted == 4

        # -----------------------------------------------------
        # REAL ENGINE
        # -----------------------------------------------------
        for i in range(cycles):
            expert_id = i % 64

            requests = scheduler.plan_prefetch_predictions(
                [expert_id]
            )

            assert len(requests) == 4

            started_body = time.perf_counter_ns()

            submitted_experts: list[int] = []

            for request in requests:
                task = _make_task_from_request(
                    real_pipeline,
                    request,
                )

                started = time.perf_counter_ns()

                decision = admission.evaluate(
                    task,
                    real_engine,
                )

                real_admission_samples.append(
                    time.perf_counter_ns() - started
                )

                assert decision.admitted

                started = time.perf_counter_ns()

                assert real_engine.submit(task)

                real_submit_samples.append(
                    time.perf_counter_ns() - started
                )

                submitted_experts.append(
                    task.expert_id
                )

            real_body_samples.append(
                time.perf_counter_ns() - started_body
            )

            assert len(submitted_experts) == 4

            # Wait only after the timed body.
            for submitted_expert in submitted_experts:
                deadline = time.monotonic() + 5.0

                while time.monotonic() < deadline:
                    status = real_engine.task_status(
                        submitted_expert
                    )

                    if (
                        status is not None
                        and status.name == "COMPLETED"
                    ):
                        break

                    if (
                        status is not None
                        and status.name in (
                            "FAILED",
                            "CANCELLED",
                        )
                    ):
                        raise AssertionError(
                            f"task failed: {status}"
                        )

                    time.sleep(0.00001)
                else:
                    raise AssertionError(
                        f"task {submitted_expert} "
                        "did not complete"
                    )

        _report(
            "7-F-15 NullEngine live process body",
            null_body_samples,
        )

        _report(
            "7-F-15 RealEngine live process body",
            real_body_samples,
        )

        _report(
            "7-F-15 NullEngine admission",
            null_admission_samples,
        )

        _report(
            "7-F-15 RealEngine admission",
            real_admission_samples,
        )

        _report(
            "7-F-15 NullEngine submit",
            null_submit_samples,
        )

        _report(
            "7-F-15 RealEngine submit",
            real_submit_samples,
        )

        real_median = median(
            [x / 1_000.0 for x in real_body_samples]
        )

        null_median = median(
            [x / 1_000.0 for x in null_body_samples]
        )

        print()
        print("7-F-15 RealEngine - NullEngine delta")
        print("------------------------------------")
        print(
            f"median delta: "
            f"{real_median - null_median:.3f} us"
        )

    finally:
        real_pipeline.stop()


if __name__ == "__main__":
    test_7f15_pipeline_live_engine_delta()
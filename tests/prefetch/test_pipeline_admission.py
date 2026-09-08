from __future__ import annotations

import threading

from predictive_cache.prefetch import (
    PrefetchEngine,
    PrefetchPipeline,
    PrefetchSource,
    PrefetchTarget,
)
from predictive_cache.prefetch.admission import AdmissionController
from predictive_cache.scheduler import ExpertScheduler


class FakeScheduler:
    def __init__(self, requests):
        self.requests = requests

    def plan_prefetch(self, current_experts):
        return list(self.requests)


class Request:
    def __init__(
        self,
        expert_id: int,
        *,
        priority: float,
        score: float,
    ):
        self.expert_id = expert_id
        self.priority = priority
        self.score = score


def test_pipeline_applies_admission_before_engine() -> None:
    loaded: list[int] = []
    loaded_event = threading.Event()

    def loader(task):
        loaded.append(task.expert_id)

        if len(loaded) == 1:
            loaded_event.set()

    scheduler = FakeScheduler(
        [
            Request(
                1,
                priority=0.9,
                score=0.95,
            ),
            Request(
                2,
                priority=0.2,
                score=0.95,
            ),
            Request(
                3,
                priority=0.9,
                score=0.4,
            ),
        ]
    )

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )

    admission = AdmissionController(
        min_priority=0.5,
        min_confidence=0.8,
    )

    pipeline = PrefetchPipeline(
        scheduler=scheduler,
        engine=engine,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
        admission=admission,
    )

    try:
        result = pipeline.process([100])

        # Scheduler produced three requests.
        assert len(result.scheduled_requests) == 3

        # Only request #1 passes admission.
        assert len(result.submitted_tasks) == 1
        assert result.submitted_tasks[0].expert_id == 1

        assert loaded_event.wait(timeout=2.0)

    finally:
        pipeline.stop()

    assert loaded == [1]


def test_pipeline_rejects_duplicate_before_engine_submission() -> None:
    loaded: list[int] = []

    started = threading.Event()
    release = threading.Event()

    def loader(task):
        loaded.append(task.expert_id)
        started.set()
        release.wait(timeout=2.0)

    scheduler = FakeScheduler(
        [
            Request(
                10,
                priority=1.0,
                score=1.0,
            ),
        ]
    )

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )

    admission = AdmissionController()

    pipeline = PrefetchPipeline(
        scheduler=scheduler,
        engine=engine,
        admission=admission,
    )

    try:
        # First process submits task 10.
        result1 = pipeline.process([1])

        assert len(result1.submitted_tasks) == 1
        assert started.wait(timeout=2.0)

        # Second process sees expert 10 as RUNNING.
        result2 = pipeline.process([1])

        assert len(result2.scheduled_requests) == 1
        assert len(result2.submitted_tasks) == 0

        assert loaded == [10]

    finally:
        release.set()
        pipeline.stop()
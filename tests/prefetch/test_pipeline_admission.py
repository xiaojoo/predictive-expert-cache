from __future__ import annotations

import threading
import time

from predictive_cache.prefetch import (
    PrefetchEngine,
    PrefetchPipeline,
    PrefetchSource,
    PrefetchStatus,
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

def test_pipeline_preserves_prediction_score_into_admission() -> None:
    scheduler = FakeScheduler(
        [
            Request(
                42,
                priority=0.9,
                score=0.91,
            ),
        ]
    )

    engine = PrefetchEngine(
        lambda task: None,
        num_workers=1,
    )

    admission = AdmissionController(
        min_priority=0.5,
        min_confidence=0.9,
    )

    pipeline = PrefetchPipeline(
        scheduler=scheduler,
        engine=engine,
        admission=admission,
    )

    try:
        result = pipeline.process([100])

        assert len(result.scheduled_requests) == 1
        assert len(result.admission_decisions) == 1
        assert result.admission_decisions[0].admitted is True

        assert len(result.submitted_tasks) == 1

        task = result.submitted_tasks[0]
        assert task.expert_id == 42
        assert task.priority == 0.9
        assert task.confidence == 0.91
    finally:
        pipeline.stop()

def test_pipeline_rejects_prediction_when_score_below_admission_threshold() -> None:
    scheduler = FakeScheduler(
        [
            Request(
                43,
                priority=0.9,
                score=0.89,
            ),
        ]
    )

    engine = PrefetchEngine(
        lambda task: None,
        num_workers=1,
    )

    admission = AdmissionController(
        min_priority=0.5,
        min_confidence=0.9,
    )

    pipeline = PrefetchPipeline(
        scheduler=scheduler,
        engine=engine,
        admission=admission,
    )

    try:
        result = pipeline.process([100])

        assert len(result.scheduled_requests) == 1
        assert len(result.admission_decisions) == 1

        decision = result.admission_decisions[0]
        assert decision.admitted is False

        assert result.submitted_tasks == []
        assert pipeline.queue_size == 0
    finally:
        pipeline.stop()

def test_pipeline_admission_preserves_top_k_order_after_rejection() -> None:
    scheduler = FakeScheduler(
        [
            Request(41, priority=0.9, score=0.95),
            Request(42, priority=0.9, score=0.80),
            Request(43, priority=0.9, score=0.92),
        ]
    )

    engine = PrefetchEngine(
        lambda task: None,
        num_workers=1,
    )

    admission = AdmissionController(
        min_priority=0.5,
        min_confidence=0.9,
    )

    pipeline = PrefetchPipeline(
        scheduler=scheduler,
        engine=engine,
        admission=admission,
    )

    try:
        result = pipeline.process([100])

        assert [task.expert_id for task in result.submitted_tasks] == [41, 43]
        assert [task.confidence for task in result.submitted_tasks] == [0.95, 0.92]
    finally:
        pipeline.stop()

def test_pipeline_admission_does_not_expand_top_k_after_rejection() -> None:
    scheduler = FakeScheduler(
        [
            Request(41, priority=0.9, score=0.95),
            Request(42, priority=0.9, score=0.80),
            Request(43, priority=0.9, score=0.92),
        ]
    )

    engine = PrefetchEngine(
        lambda task: None,
        num_workers=1,
    )

    admission = AdmissionController(
        min_priority=0.5,
        min_confidence=0.9,
    )

    pipeline = PrefetchPipeline(
        scheduler=scheduler,
        engine=engine,
        admission=admission,
    )

    try:
        result = pipeline.process([100])

        assert [task.expert_id for task in result.submitted_tasks] == [41, 43]
        assert len(result.submitted_tasks) == 2
        assert all(
            task.expert_id in {41, 43}
            for task in result.submitted_tasks
        )
    finally:
        pipeline.stop()

def test_pipeline_admission_accepted_task_reaches_prefetch_queue() -> None:
    loaded: list[int] = []
    started = threading.Event()
    release = threading.Event()

    def loader(task):
        loaded.append(task.expert_id)
        if task.expert_id == 1:
            started.set()
            release.wait(timeout=2.0)

    scheduler = FakeScheduler(
        [
            Request(
                1,
                priority=1.0,
                score=0.95,
            ),
            Request(
                2,
                priority=1.0,
                score=0.92,
            ),
            Request(
                3,
                priority=1.0,
                score=0.40,
            ),
        ]
    )

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )

    admission = AdmissionController(
        min_confidence=0.90,
    )

    pipeline = PrefetchPipeline(
        scheduler=scheduler,
        engine=engine,
        admission=admission,
    )

    try:
        result = pipeline.process([100])

        assert len(result.submitted_tasks) == 2
        assert [task.expert_id for task in result.submitted_tasks] == [1, 2]

        assert started.wait(timeout=2.0)

        # Task #1 is being executed.
        assert engine.running_tasks == [1]

        # Task #2 has successfully reached the Prefetch Queue.
        assert engine.queued_tasks == [2]

        # Task #3 was rejected and never entered the queue.
        assert 3 not in engine.queued_tasks
        assert 3 not in engine.running_tasks
        assert 3 not in loaded

    finally:
        release.set()
        pipeline.stop()

def test_pipeline_admission_preserves_queue_boundary_after_rejection() -> None:
    started = threading.Event()
    release = threading.Event()
    loaded: list[int] = []

    def loader(task):
        loaded.append(task.expert_id)

        if task.expert_id == 41:
            started.set()
            release.wait(timeout=2.0)

    scheduler = FakeScheduler(
        [
            Request(
                41,
                priority=1.0,
                score=0.95,
            ),
            Request(
                42,
                priority=1.0,
                score=0.92,
            ),
            Request(
                43,
                priority=1.0,
                score=0.40,
            ),
        ]
    )

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )

    admission = AdmissionController(
        min_confidence=0.90,
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

        assert len(result.scheduled_requests) == 3
        assert [task.expert_id for task in result.submitted_tasks] == [41, 42]

        assert started.wait(timeout=2.0)

        # 41 is executing, 42 is waiting in the queue.
        assert engine.running_tasks == [41]
        assert engine.queued_tasks == [42]

        # Rejected 43 must never enter the execution path.
        assert 43 not in engine.running_tasks
        assert 43 not in engine.queued_tasks
        assert 43 not in loaded

        # Admission rejection must not trigger a replacement request.
        assert len(result.submitted_tasks) == 2

    finally:
        release.set()
        pipeline.stop()

def test_pipeline_admission_accepted_task_is_completed_by_worker() -> None:
    loaded: list[int] = []
    loaded_event = threading.Event()

    def loader(task):
        loaded.append(task.expert_id)
        loaded_event.set()

    scheduler = FakeScheduler(
        [
            Request(
                51,
                priority=1.0,
                score=0.95,
            ),
        ]
    )

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )
    admission = AdmissionController(
        min_confidence=0.90,
    )

    pipeline = PrefetchPipeline(
        scheduler=scheduler,
        engine=engine,
        admission=admission,
    )

    try:
        result = pipeline.process([100])

        assert len(result.scheduled_requests) == 1
        assert len(result.submitted_tasks) == 1
        assert result.submitted_tasks[0].expert_id == 51

        assert loaded_event.wait(timeout=2.0)
    finally:
        pipeline.stop()

    assert loaded == [51]

    results = pipeline.results()
    assert len(results) == 1
    assert results[0].expert_id == 51
    assert results[0].status == PrefetchStatus.COMPLETED

def test_pipeline_admission_accepted_task_failure_reaches_failed_result() -> None:
    failed_event = threading.Event()

    def loader(task):
        failed_event.set()
        raise RuntimeError(f"load failed: {task.expert_id}")

    scheduler = FakeScheduler(
        [
            Request(
                52,
                priority=1.0,
                score=0.95,
            ),
        ]
    )

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )
    admission = AdmissionController(
        min_confidence=0.90,
    )

    pipeline = PrefetchPipeline(
        scheduler=scheduler,
        engine=engine,
        admission=admission,
    )

    try:
        result = pipeline.process([100])

        assert len(result.submitted_tasks) == 1
        assert result.submitted_tasks[0].expert_id == 52

        assert failed_event.wait(timeout=1.0)

        results = pipeline.results()
        assert len(results) == 1
        assert results[0].expert_id == 52
        assert results[0].status == PrefetchStatus.FAILED
        assert results[0].error == "load failed: 52"
    finally:
        pipeline.stop()

def test_pipeline_propagates_admission_decisions_to_stats() -> None:
    loaded = []
    loaded_event = threading.Event()

    def loader(task):
        loaded.append(task.expert_id)
        loaded_event.set()

    scheduler = FakeScheduler(
        [
            Request(61, priority=1.0, score=0.95),
            Request(62, priority=1.0, score=0.40),
        ]
    )

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )
    admission = AdmissionController(
        min_confidence=0.90,
    )

    pipeline = PrefetchPipeline(
        scheduler=scheduler,
        engine=engine,
        admission=admission,
    )

    try:
        result = pipeline.process([100])

        assert len(result.scheduled_requests) == 2
        assert [task.expert_id for task in result.submitted_tasks] == [61]

        assert loaded_event.wait(timeout=1.0)
        assert loaded == [61]

        stats = result.admission_stats
        assert stats.total == 2
        assert stats.accepted == 1
        assert stats.rejected == 1
        assert stats.reject_low_confidence == 1
    finally:
        pipeline.stop()

def test_pipeline_admission_rejection_does_not_produce_prefetch_result() -> None:
    loaded = []

    def loader(task):
        loaded.append(task.expert_id)

    scheduler = FakeScheduler(
        [
            Request(71, priority=1.0, score=0.95),
            Request(72, priority=1.0, score=0.40),
        ]
    )

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )
    admission = AdmissionController(
        min_confidence=0.90,
    )

    pipeline = PrefetchPipeline(
        scheduler=scheduler,
        engine=engine,
        admission=admission,
    )

    try:
        result = pipeline.process([100])

        assert [task.expert_id for task in result.submitted_tasks] == [71]

        # 等待队列排空并确认 accepted task 完成
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if len(pipeline.results()) == 1:
                break
            time.sleep(0.01)

        results = pipeline.results()

        assert len(results) == 1
        assert results[0].expert_id == 71
        assert results[0].status == PrefetchStatus.COMPLETED

        assert loaded == [71]
        assert 72 not in loaded
    finally:
        pipeline.stop()
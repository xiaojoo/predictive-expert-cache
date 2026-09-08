import threading

from predictive_cache.prefetch.admission import (
    AdmissionController,
    AdmissionReason,
)
from predictive_cache.prefetch.engine import PrefetchEngine
from predictive_cache.prefetch.types import (
    PrefetchSource,
    PrefetchTarget,
    PrefetchTask,
)


def make_task(
    expert_id: int = 1,
    *,
    priority: float = 1.0,
    confidence: float = 1.0,
) -> PrefetchTask:
    return PrefetchTask(
        expert_id=expert_id,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
        priority=priority,
        confidence=confidence,
    )


def make_engine(loader=None, *, max_queue_size=4):
    if loader is None:
        loader = lambda _task: None

    return PrefetchEngine(
        loader,
        max_queue_size=max_queue_size,
    )


def test_normal_task_returns_accept():
    engine = make_engine()
    admission = AdmissionController()

    decision = admission.evaluate(
        make_task(),
        engine,
    )

    assert decision.admitted is True
    assert decision.accepted is True
    assert decision.reason is AdmissionReason.ACCEPT


def test_low_priority_returns_reason():
    engine = make_engine()
    admission = AdmissionController(min_priority=0.8)

    decision = admission.evaluate(
        make_task(priority=0.5),
        engine,
    )

    assert decision.admitted is False
    assert decision.reason is AdmissionReason.REJECT_LOW_PRIORITY


def test_low_confidence_returns_reason():
    engine = make_engine()
    admission = AdmissionController(min_confidence=0.8)

    decision = admission.evaluate(
        make_task(confidence=0.5),
        engine,
    )

    assert decision.admitted is False
    assert decision.reason is AdmissionReason.REJECT_LOW_CONFIDENCE


def test_queued_duplicate_returns_reason():
    engine = make_engine(max_queue_size=4)
    admission = AdmissionController()

    task = make_task(expert_id=7)

    engine.start()

    try:
        assert engine.submit(task) is True

        decision = admission.evaluate(
            make_task(expert_id=7),
            engine,
        )

        assert decision.admitted is False
        assert decision.reason is AdmissionReason.REJECT_DUPLICATE_QUEUED
    finally:
        engine.stop(wait=True)


def test_running_duplicate_returns_reason():
    started = threading.Event()
    release = threading.Event()

    def loader(_task):
        started.set()
        release.wait(timeout=5)

    engine = make_engine(
        loader,
        max_queue_size=4,
    )
    admission = AdmissionController()

    engine.start()

    try:
        task = make_task(expert_id=8)

        assert engine.submit(task) is True
        assert started.wait(timeout=2)

        decision = admission.evaluate(
            make_task(expert_id=8),
            engine,
        )

        assert decision.admitted is False
        assert decision.reason is AdmissionReason.REJECT_DUPLICATE_RUNNING
    finally:
        release.set()
        engine.stop(wait=True)


def test_queue_full_returns_reason():
    started = threading.Event()
    release = threading.Event()

    def loader(_task):
        started.set()
        release.wait(timeout=5)

    engine = make_engine(
        loader,
        max_queue_size=1,
    )
    admission = AdmissionController()

    engine.start()

    try:
        first = make_task(expert_id=1)
        second = make_task(expert_id=2)
        third = make_task(expert_id=3)

        assert engine.submit(first) is True
        assert started.wait(timeout=2)

        assert engine.submit(second) is True

        decision = admission.evaluate(
            third,
            engine,
        )

        assert decision.admitted is False
        assert decision.reason is AdmissionReason.REJECT_QUEUE_FULL
    finally:
        release.set()
        engine.stop(wait=True)


def test_admit_remains_backward_compatible():
    engine = make_engine()
    admission = AdmissionController(min_priority=0.8)

    assert admission.admit(
        make_task(priority=1.0),
        engine,
    ) is True

    assert admission.admit(
        make_task(priority=0.5),
        engine,
    ) is False
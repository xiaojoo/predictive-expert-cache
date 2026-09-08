import pytest

from predictive_cache.prefetch import (
    PrefetchEngine,
    PrefetchSource,
    PrefetchTarget,
    PrefetchTask,
)
from predictive_cache.prefetch.admission import AdmissionController


def make_task(
    expert_id: int = 41,
    *,
    priority: float = 1.0,
    confidence: float = 0.9,
) -> PrefetchTask:
    return PrefetchTask(
        expert_id=expert_id,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
        priority=priority,
        confidence=confidence,
    )


def make_engine(
    *,
    max_queue_size: int | None = None,
) -> PrefetchEngine:
    return PrefetchEngine(
        lambda task: None,
        max_queue_size=max_queue_size,
    )


def test_normal_task_is_admitted() -> None:
    engine = make_engine()
    engine.start()

    try:
        controller = AdmissionController(
            min_priority=0.5,
            min_confidence=0.8,
        )

        task = make_task(
            priority=0.7,
            confidence=0.9,
        )

        assert controller.admit(task, engine) is True

    finally:
        engine.stop()


def test_low_priority_task_is_rejected() -> None:
    engine = make_engine()
    engine.start()

    try:
        controller = AdmissionController(
            min_priority=0.5,
            min_confidence=0.8,
        )

        task = make_task(
            priority=0.4,
            confidence=0.9,
        )

        assert controller.admit(task, engine) is False

    finally:
        engine.stop()


def test_low_confidence_task_is_rejected() -> None:
    engine = make_engine()
    engine.start()

    try:
        controller = AdmissionController(
            min_priority=0.5,
            min_confidence=0.8,
        )

        task = make_task(
            priority=0.9,
            confidence=0.7,
        )

        assert controller.admit(task, engine) is False

    finally:
        engine.stop()


def test_duplicate_queued_task_is_rejected() -> None:
    engine = make_engine()
    engine.start()

    try:
        controller = AdmissionController()

        task = make_task(expert_id=41)

        assert engine.submit(task) is True
        assert 41 in engine.queued_tasks

        duplicate = make_task(
            expert_id=41,
            priority=2.0,
            confidence=1.0,
        )

        assert controller.admit(duplicate, engine) is False

    finally:
        engine.stop()


def test_duplicate_running_task_is_rejected() -> None:
    started = False

    def loader(task):
        nonlocal started
        started = True

        import threading

        event.wait(timeout=2.0)

    import threading

    event = threading.Event()

    engine = PrefetchEngine(loader)
    engine.start()

    try:
        controller = AdmissionController()

        task = make_task(expert_id=41)

        assert engine.submit(task) is True

        deadline = __import__("time").time() + 2.0

        while __import__("time").time() < deadline:
            if started:
                break

            __import__("time").sleep(0.01)

        assert started is True
        assert 41 in engine.running_tasks

        duplicate = make_task(
            expert_id=41,
            priority=2.0,
            confidence=1.0,
        )

        assert controller.admit(duplicate, engine) is False

    finally:
        event.set()
        engine.stop()


def test_full_queue_rejects_task() -> None:
    engine = make_engine(max_queue_size=1)
    engine.start()

    try:
        controller = AdmissionController()

        first = make_task(expert_id=41)

        assert engine.submit(first) is True
        assert engine.queue_full is True

        second = make_task(expert_id=42)

        assert controller.admit(second, engine) is False

    finally:
        engine.stop()


def test_completed_task_is_not_considered_duplicate() -> None:
    import time

    engine = make_engine()
    engine.start()

    try:
        controller = AdmissionController()

        task = make_task(expert_id=41)

        assert engine.submit(task) is True

        deadline = time.time() + 2.0

        while time.time() < deadline:
            if engine.completed_tasks == [41]:
                break

            time.sleep(0.01)

        assert 41 in engine.completed_tasks

        next_task = make_task(expert_id=41)

        assert controller.admit(next_task, engine) is True

    finally:
        engine.stop()


def test_negative_min_priority_is_rejected() -> None:
    with pytest.raises(ValueError, match="min_priority"):
        AdmissionController(min_priority=-0.1)


def test_negative_min_confidence_is_rejected() -> None:
    with pytest.raises(ValueError, match="min_confidence"):
        AdmissionController(min_confidence=-0.1)


def test_confidence_above_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="min_confidence"):
        AdmissionController(min_confidence=1.1)
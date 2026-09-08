from __future__ import annotations

import threading
import time

from predictive_cache.prefetch import (
    PrefetchEngine,
    PrefetchSource,
    PrefetchStatus,
    PrefetchTarget,
    PrefetchTask,
)
from predictive_cache.prefetch.admission import AdmissionController


def make_task(
    expert_id: int,
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


def wait_for_status(
    engine: PrefetchEngine,
    expert_id: int,
    status: PrefetchStatus,
    timeout: float = 2.0,
) -> bool:
    deadline = time.time() + timeout

    while time.time() < deadline:
        if engine.task_status(expert_id) == status:
            return True

        time.sleep(0.01)

    return False


def test_cancelled_task_can_be_admitted_again() -> None:
    started = threading.Event()
    release = threading.Event()

    def loader(task: PrefetchTask) -> None:
        started.set()
        release.wait(timeout=2.0)

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )

    engine.start()

    try:
        admission = AdmissionController()

        # Occupy worker.
        first = make_task(1)

        assert engine.submit(first) is True
        assert started.wait(timeout=2.0)

        # Keep second task queued.
        task = make_task(2)

        assert engine.submit(task) is True
        assert engine.task_status(2) == PrefetchStatus.QUEUED

        # Cancel it.
        assert engine.cancel(2) is True
        assert engine.task_status(2) == PrefetchStatus.CANCELLED

        # CANCELLED must not be considered duplicate.
        retry = make_task(2)

        assert admission.admit(retry, engine) is True

        # And the retry must actually be accepted by Engine.
        assert engine.submit(retry) is True

    finally:
        release.set()
        engine.stop()


def test_completed_task_can_be_admitted_again() -> None:
    finished = threading.Event()

    def loader(task: PrefetchTask) -> None:
        finished.set()

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )

    engine.start()

    try:
        admission = AdmissionController()

        task = make_task(10)

        assert engine.submit(task) is True
        assert finished.wait(timeout=2.0)

        assert wait_for_status(
            engine,
            10,
            PrefetchStatus.COMPLETED,
        )

        retry = make_task(10)

        assert admission.admit(
            retry,
            engine,
        ) is True

        assert engine.submit(retry) is True

    finally:
        engine.stop()


def test_failed_task_can_be_admitted_again() -> None:
    def loader(task: PrefetchTask) -> None:
        raise RuntimeError("expected failure")

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )

    engine.start()

    try:
        admission = AdmissionController()

        task = make_task(20)

        assert engine.submit(task) is True

        assert wait_for_status(
            engine,
            20,
            PrefetchStatus.FAILED,
        )

        retry = make_task(20)

        assert admission.admit(
            retry,
            engine,
        ) is True

    finally:
        engine.stop()


def test_running_task_cannot_be_re_admitted() -> None:
    started = threading.Event()
    release = threading.Event()

    def loader(task: PrefetchTask) -> None:
        started.set()
        release.wait(timeout=2.0)

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )

    engine.start()

    try:
        admission = AdmissionController()

        task = make_task(30)

        assert engine.submit(task) is True
        assert started.wait(timeout=2.0)

        assert engine.task_status(30) == PrefetchStatus.RUNNING

        duplicate = make_task(
            30,
            priority=10.0,
            confidence=1.0,
        )

        assert admission.admit(
            duplicate,
            engine,
        ) is False

    finally:
        release.set()
        engine.stop()


def test_queued_task_cannot_be_re_admitted() -> None:
    started = threading.Event()
    release = threading.Event()

    def loader(task: PrefetchTask) -> None:
        started.set()
        release.wait(timeout=2.0)

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )

    engine.start()

    try:
        admission = AdmissionController()

        first = make_task(40)

        assert engine.submit(first) is True
        assert started.wait(timeout=2.0)

        task = make_task(41)

        assert engine.submit(task) is True
        assert engine.task_status(41) == PrefetchStatus.QUEUED

        duplicate = make_task(
            41,
            priority=10.0,
            confidence=1.0,
        )

        assert admission.admit(
            duplicate,
            engine,
        ) is False

    finally:
        release.set()
        engine.stop()
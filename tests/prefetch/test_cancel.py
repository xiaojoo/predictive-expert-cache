from __future__ import annotations

import threading

from predictive_cache.prefetch import (
    PrefetchEngine,
    PrefetchSource,
    PrefetchStatus,
    PrefetchTarget,
    PrefetchTask,
)


def make_task(expert_id: int) -> PrefetchTask:
    return PrefetchTask(
        expert_id=expert_id,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
        priority=float(expert_id),
    )


def test_cancel_queued_task() -> None:
    """
    QUEUED -> CANCELLED
    """

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

    # 第一个任务占住 Worker。
    assert engine.submit(make_task(1)) is True
    assert started.wait(timeout=2.0)

    # 第二个任务保持在 QUEUED。
    assert engine.submit(make_task(2)) is True

    assert (
        engine.task_status(2)
        == PrefetchStatus.QUEUED
    )

    assert engine.cancel(2) is True

    assert (
        engine.task_status(2)
        == PrefetchStatus.CANCELLED
    )

    assert 2 in engine.cancelled_tasks
    assert 2 not in engine.queued_tasks

    # 被取消任务已经从 queue 移除。
    assert engine.queue_size == 0
    assert engine.unfinished_tasks == 1

    release.set()

    engine.stop()

    # expert 2 没有被 loader 执行。
    assert engine.results()[0].expert_id == 1


def test_cancelled_task_is_not_executed() -> None:
    """
    被取消任务不能进入 loader。
    """

    loaded: list[int] = []

    started = threading.Event()
    release = threading.Event()

    def loader(task: PrefetchTask) -> None:
        loaded.append(task.expert_id)

        if task.expert_id == 1:
            started.set()
            release.wait(timeout=2.0)

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )

    engine.start()

    assert engine.submit(make_task(1)) is True

    assert started.wait(timeout=2.0)

    assert engine.submit(make_task(2)) is True
    assert engine.submit(make_task(3)) is True

    assert engine.cancel(2) is True
    assert engine.cancel(3) is True

    release.set()

    engine.stop()

    assert loaded == [1]

    assert (
        engine.task_status(2)
        == PrefetchStatus.CANCELLED
    )

    assert (
        engine.task_status(3)
        == PrefetchStatus.CANCELLED
    )


def test_cancel_running_task_is_rejected() -> None:
    """
    RUNNING 任务不能被强制取消。

    这是当前阶段的设计约束。
    """

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

    assert engine.submit(make_task(10)) is True

    assert started.wait(timeout=2.0)

    assert (
        engine.task_status(10)
        == PrefetchStatus.RUNNING
    )

    assert engine.cancel(10) is False

    assert (
        engine.task_status(10)
        == PrefetchStatus.RUNNING
    )

    release.set()

    engine.stop()

    assert (
        engine.task_status(10)
        == PrefetchStatus.COMPLETED
    )


def test_cancel_unknown_task() -> None:
    """
    未知任务不能取消。
    """

    engine = PrefetchEngine(
        lambda task: None,
        num_workers=1,
    )

    assert engine.cancel(999) is False


def test_cancel_completed_task() -> None:
    """
    COMPLETED 任务不能再次取消。
    """

    finished = threading.Event()

    def loader(task: PrefetchTask) -> None:
        finished.set()

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )

    engine.start()

    assert engine.submit(make_task(20)) is True

    assert finished.wait(timeout=2.0)

    engine.stop()

    assert (
        engine.task_status(20)
        == PrefetchStatus.COMPLETED
    )

    assert engine.cancel(20) is False

    assert (
        engine.task_status(20)
        == PrefetchStatus.COMPLETED
    )


def test_cancel_failed_task() -> None:
    """
    FAILED 任务不能再次取消。
    """

    def loader(task: PrefetchTask) -> None:
        raise RuntimeError("expected failure")

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )

    engine.start()

    assert engine.submit(make_task(30)) is True

    engine.stop()

    assert (
        engine.task_status(30)
        == PrefetchStatus.FAILED
    )

    assert engine.cancel(30) is False


def test_cancel_reduces_unfinished_tasks() -> None:
    """
    cancel() 必须正确减少 unfinished_tasks。

    否则后面的 stop().join() 会永久等待。
    """

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

    assert engine.submit(make_task(1)) is True
    assert started.wait(timeout=2.0)

    assert engine.submit(make_task(2)) is True
    assert engine.submit(make_task(3)) is True

    assert engine.unfinished_tasks == 3

    assert engine.cancel(2) is True
    assert engine.unfinished_tasks == 2

    assert engine.cancel(3) is True
    assert engine.unfinished_tasks == 1

    release.set()

    engine.stop()

    assert engine.unfinished_tasks == 0


def test_cancel_multiple_queued_tasks() -> None:
    """
    多个 queued task 可以分别取消。
    """

    started = threading.Event()
    release = threading.Event()

    loaded: list[int] = []

    def loader(task: PrefetchTask) -> None:
        loaded.append(task.expert_id)

        if task.expert_id == 1:
            started.set()
            release.wait(timeout=2.0)

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )

    engine.start()

    assert engine.submit(make_task(1)) is True
    assert started.wait(timeout=2.0)

    for expert_id in range(2, 7):
        assert engine.submit(make_task(expert_id)) is True

    assert engine.queue_size == 5

    assert engine.cancel(2) is True
    assert engine.cancel(4) is True
    assert engine.cancel(6) is True

    assert engine.queue_size == 2

    release.set()

    engine.stop()

    assert loaded == [1, 5, 3]

    assert engine.task_status(2) == PrefetchStatus.CANCELLED
    assert engine.task_status(4) == PrefetchStatus.CANCELLED
    assert engine.task_status(6) == PrefetchStatus.CANCELLED

    assert engine.task_status(3) == PrefetchStatus.COMPLETED
    assert engine.task_status(5) == PrefetchStatus.COMPLETED


def test_shutdown_after_cancellation() -> None:
    """
    Cancellation 后仍然可以正常 graceful shutdown。
    """

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

    assert engine.submit(make_task(1)) is True
    assert started.wait(timeout=2.0)

    assert engine.submit(make_task(2)) is True
    assert engine.submit(make_task(3)) is True

    assert engine.cancel(2) is True
    assert engine.cancel(3) is True

    release.set()

    engine.stop()

    assert engine.running is False
    assert engine.queue_size == 0
    assert engine.unfinished_tasks == 0

    assert engine.task_status(1) == PrefetchStatus.COMPLETED
    assert engine.task_status(2) == PrefetchStatus.CANCELLED
    assert engine.task_status(3) == PrefetchStatus.CANCELLED
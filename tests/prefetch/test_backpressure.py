from __future__ import annotations

import threading

from predictive_cache.prefetch import (
    PrefetchEngine,
    PrefetchSource,
    PrefetchStatus,
    PrefetchTarget,
    PrefetchTask,
)


def make_task(
    expert_id: int,
    priority: float = 0.0,
) -> PrefetchTask:
    return PrefetchTask(
        expert_id=expert_id,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
        priority=priority,
    )


def test_queue_rejects_when_full() -> None:
    started = threading.Event()
    release = threading.Event()

    def loader(task: PrefetchTask) -> None:
        started.set()
        release.wait(timeout=2.0)

    engine = PrefetchEngine(
        loader,
        num_workers=1,
        max_queue_size=2,
    )

    engine.start()

    # RUNNING
    assert engine.submit(make_task(1, 0.0))
    assert started.wait(timeout=2.0)

    # QUEUED
    assert engine.submit(make_task(2, 0.0))
    assert engine.submit(make_task(3, 0.0))

    assert engine.queue_size == 2
    assert engine.queue_full is True

    # Queue 已满
    assert engine.submit(make_task(4, 0.0)) is False

    assert (
        engine.task_status(4)
        is None
    )

    release.set()
    engine.stop()


def test_queue_accepts_after_capacity_is_released() -> None:
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
        max_queue_size=1,
    )

    engine.start()

    assert engine.submit(make_task(1, 0.0))
    assert started.wait(timeout=2.0)

    # Queue capacity = 1
    assert engine.submit(make_task(2, 0.0))
    assert engine.submit(make_task(3, 0.0)) is False

    release.set()

    # 等待第一个 queued task 被取出
    while engine.queue_size != 0:
        threading.Event().wait(0.01)

    # 此时可以继续提交
    assert engine.submit(make_task(3, 0.0))

    engine.stop()

    assert 3 in loaded


def test_queue_full_does_not_create_task_state() -> None:
    started = threading.Event()
    release = threading.Event()

    def loader(task: PrefetchTask) -> None:
        started.set()
        release.wait(timeout=2.0)

    engine = PrefetchEngine(
        loader,
        num_workers=1,
        max_queue_size=1,
    )

    engine.start()

    assert engine.submit(make_task(1, 0.0))
    assert started.wait(timeout=2.0)

    assert engine.submit(make_task(2, 0.0))

    # Queue 满，expert 3 被拒绝。
    assert engine.submit(make_task(3, 0.0)) is False

    assert engine.task_status(3) is None

    assert 3 not in engine.queued_tasks
    assert 3 not in engine.running_tasks

    release.set()
    engine.stop()


def test_cancel_frees_backpressure_capacity() -> None:
    started = threading.Event()
    release = threading.Event()

    def loader(task: PrefetchTask) -> None:
        started.set()
        release.wait(timeout=2.0)

    engine = PrefetchEngine(
        loader,
        num_workers=1,
        max_queue_size=1,
    )

    engine.start()

    assert engine.submit(make_task(1, 0.0))
    assert started.wait(timeout=2.0)

    assert engine.submit(make_task(2, 0.0))

    assert engine.queue_full is True

    # 取消 queued task
    assert engine.cancel(2) is True

    assert engine.queue_full is False
    assert engine.queue_size == 0

    # 容量重新释放
    assert engine.submit(make_task(3, 0.0)) is True

    release.set()

    engine.stop()

    assert (
        engine.task_status(2)
        == PrefetchStatus.CANCELLED
    )

    assert (
        engine.task_status(3)
        == PrefetchStatus.COMPLETED
    )


def test_unbounded_queue_remains_supported() -> None:
    """
    max_queue_size=None 保持旧行为。
    """

    engine = PrefetchEngine(
        lambda task: None,
        num_workers=1,
        max_queue_size=None,
    )

    assert engine.max_queue_size is None
    assert engine.queue_full is False

    engine.start()

    for expert_id in range(20):
        assert engine.submit(
            make_task(expert_id)
        )

    engine.stop()

    assert len(engine.results()) == 20


def test_invalid_max_queue_size() -> None:
    try:
        PrefetchEngine(
            lambda task: None,
            max_queue_size=0,
        )
        assert False
    except ValueError:
        pass

    try:
        PrefetchEngine(
            lambda task: None,
            max_queue_size=-1,
        )
        assert False
    except ValueError:
        pass
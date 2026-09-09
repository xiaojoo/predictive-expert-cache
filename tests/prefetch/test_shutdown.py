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


def make_task(expert_id: int) -> PrefetchTask:
    return PrefetchTask(
        expert_id=expert_id,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
        priority=float(expert_id),
    )


def test_shutdown_empty_queue() -> None:
    """
    空队列也应该能够正常 graceful shutdown。
    """

    engine = PrefetchEngine(
        lambda task: None,
        num_workers=1,
    )

    engine.start()

    assert engine.running is True

    engine.stop()

    assert engine.running is False
    assert engine.queue_size == 0
    assert engine.unfinished_tasks == 0


def test_shutdown_drains_queue() -> None:
    """
    shutdown 时不能丢掉已经进入 queue 的任务。

    3 个任务全部应该执行完成。
    """

    loaded: list[int] = []

    def loader(task: PrefetchTask) -> None:
        loaded.append(task.expert_id)

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )

    engine.start()

    assert engine.submit(make_task(1)) is True
    assert engine.submit(make_task(2)) is True
    assert engine.submit(make_task(3)) is True

    engine.stop()

    assert set(loaded) == {1, 2, 3}

    assert engine.running is False
    assert engine.queue_size == 0
    assert engine.unfinished_tasks == 0

    results = engine.results()

    assert len(results) == 3

    assert all(
        result.status == PrefetchStatus.COMPLETED
        for result in results
    )


def test_submit_rejected_after_shutdown() -> None:
    """
    shutdown 之后不能再提交新任务。
    """

    loaded: list[int] = []

    engine = PrefetchEngine(
        lambda task: loaded.append(task.expert_id),
        num_workers=1,
    )

    engine.start()

    engine.stop()

    assert engine.submit(make_task(1)) is False

    assert loaded == []


def test_shutdown_is_idempotent() -> None:
    """
    stop() 可以重复调用。
    """

    engine = PrefetchEngine(
        lambda task: None,
        num_workers=1,
    )

    engine.start()

    engine.stop()

    # 第二次 stop 不应该异常。
    engine.stop()

    assert engine.running is False


def test_shutdown_waits_for_running_task() -> None:
    """
    shutdown 必须等待当前正在执行的任务完成。
    """

    started = threading.Event()
    finished = threading.Event()

    def loader(task: PrefetchTask) -> None:
        started.set()

        # 模拟一个正在执行的慢任务。
        time.sleep(0.2)

        finished.set()

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )

    engine.start()

    assert engine.submit(make_task(1)) is True

    # 确保任务已经进入 loader。
    assert started.wait(timeout=2.0)

    # stop(wait=True) 必须等待 loader 完成。
    engine.stop(wait=True)

    assert finished.is_set() is True
    assert engine.running is False
    assert engine.unfinished_tasks == 0


def test_task_failure_does_not_break_shutdown() -> None:
    """
    loader 抛异常时：

    - Worker 不能崩溃
    - task_done() 必须执行
    - shutdown 必须正常完成
    - result 状态应该是 FAILED
    """

    def loader(task: PrefetchTask) -> None:
        raise RuntimeError(
            f"failed to load expert {task.expert_id}"
        )

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )

    engine.start()

    assert engine.submit(make_task(42)) is True

    engine.stop()

    assert engine.running is False
    assert engine.queue_size == 0
    assert engine.unfinished_tasks == 0

    results = engine.results()

    assert len(results) == 1

    result = results[0]

    assert result.expert_id == 42
    assert result.status == PrefetchStatus.FAILED
    assert result.error is not None
    assert "42" in result.error


def test_shutdown_drains_multiple_workers() -> None:
    """
    多 Worker 模式下也必须保证所有任务最终完成。
    """

    loaded: list[int] = []
    lock = threading.Lock()

    def loader(task: PrefetchTask) -> None:
        with lock:
            loaded.append(task.expert_id)

        time.sleep(0.02)

    engine = PrefetchEngine(
        loader,
        num_workers=3,
    )

    engine.start()

    for expert_id in range(10):
        assert engine.submit(make_task(expert_id)) is True

    engine.stop()

    assert set(loaded) == set(range(10))

    assert len(loaded) == 10

    assert engine.queue_size == 0
    assert engine.unfinished_tasks == 0
    assert engine.running is False

    results = engine.results()

    assert len(results) == 10

def test_engine_can_restart_after_graceful_shutdown():
    loaded: list[int] = []

    def loader(task: PrefetchTask) -> None:
        loaded.append(task.expert_id)

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )

    # First lifecycle:
    # start -> submit -> completed -> stop
    engine.start()

    assert engine.running is True
    assert engine.submit(make_task(101)) is True

    engine.stop()

    assert engine.running is False
    assert engine.queue_size == 0
    assert engine.unfinished_tasks == 0
    assert engine.task_status(101) == PrefetchStatus.COMPLETED

    # Second lifecycle:
    # start -> submit -> completed -> stop
    engine.start()

    assert engine.running is True
    assert engine.submit(make_task(102)) is True

    engine.stop()

    assert engine.running is False
    assert engine.queue_size == 0
    assert engine.unfinished_tasks == 0

    assert loaded == [101, 102]

    assert engine.task_status(101) == PrefetchStatus.COMPLETED
    assert engine.task_status(102) == PrefetchStatus.COMPLETED

    results = engine.results()

    assert len(results) == 2
    assert {result.expert_id for result in results} == {101, 102}
    assert all(
        result.status == PrefetchStatus.COMPLETED
        for result in results
    )
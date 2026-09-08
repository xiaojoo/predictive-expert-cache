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


def test_task_starts_as_queued() -> None:
    """
    submit() 成功后，Task 应该进入 QUEUED。
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

    task = make_task(1)

    assert engine.submit(task) is True

    # Worker 可能已经非常快地取走任务，
    # 所以只验证最终能够进入 RUNNING。
    assert started.wait(timeout=2.0)

    assert engine.task_status(1) == PrefetchStatus.RUNNING

    release.set()

    engine.stop()

    assert engine.task_status(1) == PrefetchStatus.COMPLETED


def test_task_reaches_completed() -> None:
    """
    正常 loader：

        QUEUED -> RUNNING -> COMPLETED
    """

    finished = threading.Event()

    def loader(task: PrefetchTask) -> None:
        finished.set()

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )

    engine.start()

    assert engine.submit(make_task(10)) is True

    assert finished.wait(timeout=2.0)

    engine.stop()

    assert (
        engine.task_status(10)
        == PrefetchStatus.COMPLETED
    )

    assert 10 in engine.completed_tasks
    assert 10 not in engine.failed_tasks


def test_task_reaches_failed() -> None:
    """
    loader 异常：

        QUEUED -> RUNNING -> FAILED
    """

    def loader(task: PrefetchTask) -> None:
        raise RuntimeError("load failed")

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )

    engine.start()

    assert engine.submit(make_task(20)) is True

    engine.stop()

    assert (
        engine.task_status(20)
        == PrefetchStatus.FAILED
    )

    assert 20 in engine.failed_tasks
    assert 20 not in engine.completed_tasks


def test_running_tasks_tracks_active_task() -> None:
    """
    Task 执行期间应该出现在 running_tasks。
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

    assert engine.submit(make_task(30)) is True

    assert started.wait(timeout=2.0)

    assert 30 in engine.running_tasks

    release.set()

    engine.stop()

    assert 30 not in engine.running_tasks
    assert 30 in engine.completed_tasks


def test_multiple_task_states() -> None:
    """
    多 Worker 下可以同时存在：

        RUNNING
        COMPLETED
        FAILED
    """

    started = threading.Event()
    release = threading.Event()

    def loader(task: PrefetchTask) -> None:
        if task.expert_id == 1:
            started.set()
            release.wait(timeout=2.0)

        elif task.expert_id == 2:
            time.sleep(0.05)

        else:
            raise RuntimeError("expected failure")

    engine = PrefetchEngine(
        loader,
        num_workers=3,
    )

    engine.start()

    assert engine.submit(make_task(1)) is True
    assert engine.submit(make_task(2)) is True
    assert engine.submit(make_task(3)) is True

    assert started.wait(timeout=2.0)

    # expert 1 被阻塞，所以应该仍然 RUNNING。
    assert engine.task_status(1) == PrefetchStatus.RUNNING

    release.set()

    engine.stop()

    states = engine.task_states()

    assert states[1] == PrefetchStatus.COMPLETED
    assert states[2] == PrefetchStatus.COMPLETED
    assert states[3] == PrefetchStatus.FAILED


def test_task_status_unknown_expert() -> None:
    """
    从未提交过的 expert 返回 None。
    """

    engine = PrefetchEngine(
        lambda task: None,
        num_workers=1,
    )

    assert engine.task_status(999) is None


def test_task_states_returns_copy() -> None:
    """
    task_states() 返回副本。

    外部修改不能破坏 Engine 内部状态。
    """

    engine = PrefetchEngine(
        lambda task: None,
        num_workers=1,
    )

    engine.start()

    assert engine.submit(make_task(100)) is True

    engine.stop()

    states = engine.task_states()

    states[100] = PrefetchStatus.FAILED

    assert (
        engine.task_status(100)
        == PrefetchStatus.COMPLETED
    )


def test_clear_task_states() -> None:
    """
    clear_task_states() 只清理状态表：

    - 不清理 results
    """

    engine = PrefetchEngine(
        lambda task: None,
        num_workers=1,
    )

    engine.start()

    assert engine.submit(make_task(200)) is True

    engine.stop()

    assert engine.task_status(200) == PrefetchStatus.COMPLETED
    assert len(engine.results()) == 1

    engine.clear_task_states()

    assert engine.task_status(200) is None
    assert engine.task_states() == {}

    # results 不应该被清掉。
    assert len(engine.results()) == 1
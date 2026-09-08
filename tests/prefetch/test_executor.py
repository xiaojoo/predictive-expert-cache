import pytest

from predictive_cache.prefetch.executor import (
    InMemoryPrefetchExecutor,
)
from predictive_cache.prefetch.task import (
    PrefetchTask,
    TaskState,
)


def make_task(
    expert_id: int = 1,
    priority: float = 0.9,
    probability: float = 0.9,
) -> PrefetchTask:
    return PrefetchTask(
        expert_id=expert_id,
        priority=priority,
        probability=probability,
    )


def test_executor_executes_task():
    executor = InMemoryPrefetchExecutor()

    task = make_task(expert_id=10)

    result = executor.execute(task)

    assert result.state == TaskState.COMPLETED
    assert result.success is True
    assert result.error is None

    assert executor.is_loaded(10)
    assert 10 in executor.loaded_experts


def test_executor_records_executed_tasks():
    executor = InMemoryPrefetchExecutor()

    task1 = make_task(expert_id=1)
    task2 = make_task(expert_id=2)

    executor.execute(task1)
    executor.execute(task2)

    assert executor.executed_tasks == [task1, task2]


def test_executor_failure_is_recorded():
    def failing_loader(task: PrefetchTask) -> None:
        raise RuntimeError("load failed")

    executor = InMemoryPrefetchExecutor(
        loader=failing_loader,
    )

    task = make_task(expert_id=5)

    result = executor.execute(task)

    assert result.state == TaskState.FAILED
    assert result.success is False
    assert result.error == "load failed"

    assert executor.is_loaded(5) is False


def test_executor_clear():
    executor = InMemoryPrefetchExecutor()

    executor.execute(make_task(expert_id=1))
    executor.execute(make_task(expert_id=2))

    assert executor.is_loaded(1)
    assert executor.is_loaded(2)

    executor.clear()

    assert executor.loaded_experts == set()
    assert executor.executed_tasks == []


def test_executor_loader_is_called():
    calls: list[int] = []

    def loader(task: PrefetchTask) -> None:
        calls.append(task.expert_id)

    executor = InMemoryPrefetchExecutor(
        loader=loader,
    )

    executor.execute(make_task(expert_id=42))

    assert calls == [42]
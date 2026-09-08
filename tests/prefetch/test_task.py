import pytest

from predictive_cache.prefetch.task import (
    PrefetchTask,
    TaskResult,
    TaskState,
)


def test_create_prefetch_task():
    task = PrefetchTask(
        expert_id=10,
        priority=0.95,
        probability=0.95,
    )

    assert task.expert_id == 10
    assert task.priority == 0.95
    assert task.probability == 0.95
    assert task.source == "prediction"


def test_task_rejects_negative_expert_id():
    with pytest.raises(ValueError):
        PrefetchTask(
            expert_id=-1,
            priority=0.5,
            probability=0.5,
        )


def test_task_rejects_invalid_probability():
    with pytest.raises(ValueError):
        PrefetchTask(
            expert_id=1,
            priority=0.5,
            probability=1.5,
        )


def test_task_rejects_negative_priority():
    with pytest.raises(ValueError):
        PrefetchTask(
            expert_id=1,
            priority=-0.1,
            probability=0.5,
        )


def test_task_result_success():
    task = PrefetchTask(
        expert_id=1,
        priority=0.8,
        probability=0.8,
    )

    result = TaskResult(
        task=task,
        state=TaskState.COMPLETED,
    )

    assert result.success is True
    assert result.error is None


def test_task_result_failure():
    task = PrefetchTask(
        expert_id=1,
        priority=0.8,
        probability=0.8,
    )

    result = TaskResult(
        task=task,
        state=TaskState.FAILED,
        error="load failed",
    )

    assert result.success is False
    assert result.error == "load failed"
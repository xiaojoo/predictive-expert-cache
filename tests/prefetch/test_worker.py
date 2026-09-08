from predictive_cache.prefetch.executor import (
    InMemoryPrefetchExecutor,
)
from predictive_cache.prefetch.queue import PrefetchQueue
from predictive_cache.prefetch.task import (
    PrefetchTask,
    TaskState,
)
from predictive_cache.prefetch.worker import PrefetchWorker


def make_task(
    expert_id: int,
    priority: float,
    probability: float = 0.9,
) -> PrefetchTask:
    return PrefetchTask(
        expert_id=expert_id,
        priority=priority,
        probability=probability,
    )


def test_worker_run_once():
    queue = PrefetchQueue()
    executor = InMemoryPrefetchExecutor()

    queue.put(make_task(1, priority=0.5))

    worker = PrefetchWorker(queue, executor)

    result = worker.run_once()

    assert result is not None
    assert result.state == TaskState.COMPLETED
    assert result.task.expert_id == 1

    assert executor.is_loaded(1)
    assert queue.size() == 0


def test_worker_run_empty_queue():
    queue = PrefetchQueue()
    executor = InMemoryPrefetchExecutor()

    worker = PrefetchWorker(queue, executor)

    result = worker.run_once()

    assert result is None
    assert worker.results == []


def test_worker_run_all_tasks():
    queue = PrefetchQueue()
    executor = InMemoryPrefetchExecutor()

    queue.put(make_task(1, priority=0.3))
    queue.put(make_task(2, priority=0.9))
    queue.put(make_task(3, priority=0.6))

    worker = PrefetchWorker(queue, executor)

    results = worker.run()

    assert len(results) == 3

    assert [r.task.expert_id for r in results] == [
        2,
        3,
        1,
    ]

    assert queue.size() == 0
    assert executor.loaded_experts == {1, 2, 3}


def test_worker_respects_max_tasks():
    queue = PrefetchQueue()
    executor = InMemoryPrefetchExecutor()

    queue.put(make_task(1, priority=0.3))
    queue.put(make_task(2, priority=0.9))
    queue.put(make_task(3, priority=0.6))

    worker = PrefetchWorker(queue, executor)

    results = worker.run(max_tasks=2)

    assert len(results) == 2

    assert [r.task.expert_id for r in results] == [
        2,
        3,
    ]

    assert queue.size() == 1
    assert executor.loaded_experts == {2, 3}


def test_worker_records_results():
    queue = PrefetchQueue()
    executor = InMemoryPrefetchExecutor()

    queue.put(make_task(10, priority=0.8))
    queue.put(make_task(20, priority=0.7))

    worker = PrefetchWorker(queue, executor)

    worker.run()

    assert len(worker.results) == 2
    assert worker.results[0].task.expert_id == 10
    assert worker.results[1].task.expert_id == 20


def test_worker_clear_results():
    queue = PrefetchQueue()
    executor = InMemoryPrefetchExecutor()

    queue.put(make_task(1, priority=0.8))

    worker = PrefetchWorker(queue, executor)

    worker.run()

    assert len(worker.results) == 1

    worker.clear_results()

    assert worker.results == []


def test_worker_records_failed_result():
    queue = PrefetchQueue()

    def failing_loader(task: PrefetchTask) -> None:
        raise RuntimeError("prefetch failed")

    executor = InMemoryPrefetchExecutor(
        loader=failing_loader,
    )

    queue.put(make_task(42, priority=1.0))

    worker = PrefetchWorker(queue, executor)

    result = worker.run_once()

    assert result is not None
    assert result.state == TaskState.FAILED
    assert result.success is False
    assert result.error == "prefetch failed"

    assert queue.size() == 0
    assert worker.results == [result]
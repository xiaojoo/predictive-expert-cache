from predictive_cache.prefetch import (
    PrefetchQueue,
    PrefetchSource,
    PrefetchTarget,
    PrefetchTask,
)


def make_task(
    expert_id: int,
    priority: float,
) -> PrefetchTask:
    return PrefetchTask(
        expert_id=expert_id,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
        priority=priority,
        confidence=0.9,
    )


def test_priority_order():
    queue = PrefetchQueue()

    queue.put(make_task(41, 0.5))
    queue.put(make_task(7, 0.9))
    queue.put(make_task(88, 0.7))

    assert queue.get().expert_id == 7
    assert queue.get().expert_id == 88
    assert queue.get().expert_id == 41


def test_duplicate_expert_is_rejected():
    queue = PrefetchQueue()

    assert queue.put(make_task(41, 0.9))
    assert not queue.put(make_task(41, 0.8))

    assert queue.size() == 1


def test_contains():
    queue = PrefetchQueue()

    queue.put(make_task(41, 0.9))

    assert queue.contains(41)

    queue.get()

    assert not queue.contains(41)


def test_clear():
    queue = PrefetchQueue()

    queue.put(make_task(41, 0.9))
    queue.put(make_task(7, 0.8))

    queue.clear()

    assert queue.size() == 0

def test_queue_max_size() -> None:
    queue = PrefetchQueue(max_size=2)

    assert queue.put(make_task(1, 0.0)) is True
    assert queue.put(make_task(2, 0.0)) is True

    assert queue.size() == 2

    assert queue.put(make_task(3, 0.0)) is False

    assert queue.unfinished_tasks == 2


def test_queue_not_full_after_get() -> None:
    queue = PrefetchQueue(max_size=1)

    assert queue.put(make_task(1, 0.0)) is True

    assert queue.is_full is True

    task = queue.get()

    assert task is not None

    assert queue.is_full is False

    queue.task_done()


def test_queue_cancel_releases_capacity() -> None:
    queue = PrefetchQueue(max_size=1)

    assert queue.put(make_task(1, 0.0)) is True

    assert queue.is_full is True

    assert queue.cancel(1) is True

    assert queue.is_full is False
    assert queue.size() == 0
    assert queue.unfinished_tasks == 0

    assert queue.put(make_task(2, 0.0)) is True
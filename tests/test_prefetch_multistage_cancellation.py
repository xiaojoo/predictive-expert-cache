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
from predictive_cache.prefetch.stages import (
    PrefetchStage,
    PrefetchStageChain,
)
from predictive_cache.prefetch.transfer import (
    PrefetchTransferExecutor,
)


def _make_task(expert_id: int) -> PrefetchTask:
    return PrefetchTask(
        expert_id=expert_id,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
        priority=float(expert_id),
    )


def _make_executor(
    events: list[tuple[str, int]],
    *,
    stage_one_release: threading.Event | None = None,
) -> PrefetchTransferExecutor:
    def stage_one(task: PrefetchTask) -> None:
        events.append(("stage1", task.expert_id))

        if stage_one_release is not None:
            stage_one_release.wait(timeout=5.0)

    def stage_two(task: PrefetchTask) -> None:
        events.append(("stage2", task.expert_id))

    chain = PrefetchStageChain(
        [
            PrefetchStage(
                source=PrefetchSource.NVME,
                target=PrefetchTarget.RAM,
                handler=stage_one,
            ),
            PrefetchStage(
                source=PrefetchSource.RAM,
                target=PrefetchTarget.GPU,
                handler=stage_two,
            ),
        ]
    )

    executor = PrefetchTransferExecutor(
        lambda task: (_ for _ in ()).throw(
            AssertionError("unexpected fallback loader")
        )
    )

    executor.register(
        PrefetchSource.NVME,
        PrefetchTarget.RAM,
        chain,
    )

    return executor


def _wait_status(
    engine: PrefetchEngine,
    expert_id: int,
    expected: PrefetchStatus,
    timeout: float = 5.0,
) -> None:
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if engine.task_status(expert_id) == expected:
            return

        time.sleep(0.005)

    raise AssertionError(
        f"task {expert_id} did not reach {expected!r}; "
        f"actual={engine.task_status(expert_id)!r}"
    )


def test_queued_multistage_task_can_be_cancelled_without_running_any_stage() -> None:
    events: list[tuple[str, int]] = []

    stage_one_started = threading.Event()
    stage_one_release = threading.Event()

    def blocking_stage_one(task: PrefetchTask) -> None:
        events.append(("stage1", task.expert_id))

        if task.expert_id == 1:
            stage_one_started.set()
            stage_one_release.wait(timeout=5.0)

    def stage_two(task: PrefetchTask) -> None:
        events.append(("stage2", task.expert_id))

    chain = PrefetchStageChain(
        [
            PrefetchStage(
                source=PrefetchSource.NVME,
                target=PrefetchTarget.RAM,
                handler=blocking_stage_one,
            ),
            PrefetchStage(
                source=PrefetchSource.RAM,
                target=PrefetchTarget.GPU,
                handler=stage_two,
            ),
        ]
    )

    transfer = PrefetchTransferExecutor(
        lambda task: (_ for _ in ()).throw(
            AssertionError("fallback loader must not run")
        )
    )

    transfer.register(
        PrefetchSource.NVME,
        PrefetchTarget.RAM,
        chain,
    )

    engine = PrefetchEngine(
        lambda task: (_ for _ in ()).throw(
            AssertionError("fallback loader must not run")
        ),
        num_workers=1,
        transfer_executor=transfer,
    )

    engine.start()

    try:
        assert engine.submit(_make_task(1))
        assert stage_one_started.wait(timeout=2.0)

        assert engine.submit(_make_task(2))

        assert (
            engine.task_status(2)
            == PrefetchStatus.QUEUED
        )

        assert engine.cancel(2) is True

        assert (
            engine.task_status(2)
            == PrefetchStatus.CANCELLED
        )

        assert engine.queue_size == 0
        assert engine.unfinished_tasks == 1

        stage_one_release.set()
        engine.stop(wait=True)

        assert events == [
            ("stage1", 1),
            ("stage2", 1),
        ]

        assert engine.task_status(1) == PrefetchStatus.COMPLETED
        assert engine.task_status(2) == PrefetchStatus.CANCELLED
        assert engine.unfinished_tasks == 0

    finally:
        stage_one_release.set()

        if engine.running:
            engine.stop(wait=True)


def test_multiple_queued_multistage_tasks_can_be_selectively_cancelled() -> None:
    events: list[tuple[str, int]] = []

    stage_one_started = threading.Event()
    stage_one_release = threading.Event()

    def stage_one(task: PrefetchTask) -> None:
        events.append(("stage1", task.expert_id))

        if task.expert_id == 1:
            stage_one_started.set()
            stage_one_release.wait(timeout=5.0)

    def stage_two(task: PrefetchTask) -> None:
        events.append(("stage2", task.expert_id))

    chain = PrefetchStageChain(
        [
            PrefetchStage(
                source=PrefetchSource.NVME,
                target=PrefetchTarget.RAM,
                handler=stage_one,
            ),
            PrefetchStage(
                source=PrefetchSource.RAM,
                target=PrefetchTarget.GPU,
                handler=stage_two,
            ),
        ]
    )

    transfer = PrefetchTransferExecutor(
        lambda task: (_ for _ in ()).throw(
            AssertionError("unexpected fallback loader")
        )
    )

    transfer.register(
        PrefetchSource.NVME,
        PrefetchTarget.RAM,
        chain,
    )

    engine = PrefetchEngine(
        lambda task: (_ for _ in ()).throw(
            AssertionError("fallback loader must not run")
        ),
        num_workers=1,
        transfer_executor=transfer,
    )

    engine.start()

    try:
        assert engine.submit(_make_task(1))
        assert stage_one_started.wait(timeout=2.0)

        assert engine.submit(_make_task(2))
        assert engine.submit(_make_task(3))
        assert engine.submit(_make_task(4))

        assert engine.cancel(2) is True
        assert engine.cancel(4) is True

        assert engine.task_status(2) == PrefetchStatus.CANCELLED
        assert engine.task_status(4) == PrefetchStatus.CANCELLED

        stage_one_release.set()
        engine.stop(wait=True)

        assert engine.task_status(1) == PrefetchStatus.COMPLETED
        assert engine.task_status(3) == PrefetchStatus.COMPLETED

        assert engine.task_status(2) == PrefetchStatus.CANCELLED
        assert engine.task_status(4) == PrefetchStatus.CANCELLED

        assert events == [
            ("stage1", 1),
            ("stage2", 1),
            ("stage1", 3),
            ("stage2", 3),
        ]

        assert engine.unfinished_tasks == 0
        assert engine.queue_size == 0

    finally:
        stage_one_release.set()

        if engine.running:
            engine.stop(wait=True)


def test_cancelled_multistage_task_does_not_consume_queue_capacity() -> None:
    events: list[tuple[str, int]] = []

    stage_one_release = threading.Event()

    transfer = _make_executor(
        events,
        stage_one_release=stage_one_release,
    )

    engine = PrefetchEngine(
        lambda task: (_ for _ in ()).throw(
            AssertionError("fallback loader must not run")
        ),
        num_workers=1,
        max_queue_size=2,
        transfer_executor=transfer,
    )

    engine.start()

    try:
        assert engine.submit(_make_task(1))
        assert engine.submit(_make_task(2))
        assert engine.submit(_make_task(3)) is False

        assert engine.queue_full is True

        assert engine.cancel(2) is True

        assert engine.task_status(2) == PrefetchStatus.CANCELLED
        assert engine.queue_size == 1
        assert engine.queue_full is False

        assert engine.submit(_make_task(3)) is True

        stage_one_release.set()
        engine.stop(wait=True)

        assert engine.task_status(1) == PrefetchStatus.COMPLETED
        assert engine.task_status(2) == PrefetchStatus.CANCELLED
        assert engine.task_status(3) == PrefetchStatus.COMPLETED

        assert engine.unfinished_tasks == 0

    finally:
        stage_one_release.set()

        if engine.running:
            engine.stop(wait=True)

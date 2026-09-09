from __future__ import annotations

from threading import Event

import pytest

from predictive_cache.prefetch.stages import (
    PrefetchCancelled,
    PrefetchStage,
    PrefetchStageChain,
)
from predictive_cache.prefetch.types import (
    PrefetchSource,
    PrefetchStatus,
    PrefetchTarget,
    PrefetchTask,
)
from predictive_cache.prefetch.engine import PrefetchEngine


def _task(expert_id: int = 42) -> PrefetchTask:
    return PrefetchTask(
        expert_id=expert_id,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
        priority=1.0,
        confidence=1.0,
    )


def test_stage2_does_not_execute_after_stage1_cancellation() -> None:
    cancelled = Event()

    stage1_calls: list[int] = []
    stage2_calls: list[int] = []

    def stage1(task: PrefetchTask) -> None:
        stage1_calls.append(task.expert_id)

        # Cancellation arrives immediately after Stage 1
        # has completed.
        cancelled.set()

    def stage2(task: PrefetchTask) -> None:
        stage2_calls.append(task.expert_id)

    chain = PrefetchStageChain(
        [
            PrefetchStage(
                source=PrefetchSource.NVME,
                target=PrefetchTarget.RAM,
                handler=stage1,
            ),
            PrefetchStage(
                source=PrefetchTarget.RAM,
                target=PrefetchTarget.GPU,
                handler=stage2,
            ),
        ],
        should_cancel=cancelled.is_set,
    )

    with pytest.raises(PrefetchCancelled):
        chain.execute(_task())

    assert stage1_calls == [42]
    assert stage2_calls == []


def test_cancellation_before_first_stage_skips_all_stages() -> None:
    cancelled = Event()
    cancelled.set()

    stage1_calls: list[int] = []
    stage2_calls: list[int] = []

    chain = PrefetchStageChain(
        [
            PrefetchStage(
                source=PrefetchSource.NVME,
                target=PrefetchTarget.RAM,
                handler=lambda task: stage1_calls.append(
                    task.expert_id
                ),
            ),
            PrefetchStage(
                source=PrefetchTarget.RAM,
                target=PrefetchTarget.GPU,
                handler=lambda task: stage2_calls.append(
                    task.expert_id
                ),
            ),
        ],
        should_cancel=cancelled.is_set,
    )

    with pytest.raises(PrefetchCancelled):
        chain.execute(_task())

    assert stage1_calls == []
    assert stage2_calls == []

def test_engine_records_stage_boundary_cancellation() -> None:
    cancelled = Event()

    stage1_calls: list[int] = []
    stage2_calls: list[int] = []

    def stage1(task: PrefetchTask) -> None:
        stage1_calls.append(task.expert_id)
        cancelled.set()

    def stage2(task: PrefetchTask) -> None:
        stage2_calls.append(task.expert_id)

    chain = PrefetchStageChain(
        [
            PrefetchStage(
                source=PrefetchSource.NVME,
                target=PrefetchTarget.RAM,
                handler=stage1,
            ),
            PrefetchStage(
                source=PrefetchTarget.RAM,
                target=PrefetchTarget.GPU,
                handler=stage2,
            ),
        ],
        should_cancel=cancelled.is_set,
    )

    engine = PrefetchEngine(
        loader=lambda task: None,
        transfer_executor=chain,
    )

    task = _task(43)

    engine.start()

    try:
        assert engine.submit(task)

        engine.stop(wait=True)
    finally:
        if engine.running:
            engine.stop(wait=True)

    assert stage1_calls == [43]
    assert stage2_calls == []

    assert engine.task_status(43) == PrefetchStatus.CANCELLED
    assert engine.cancelled_tasks == [43]
    assert engine.completed_tasks == []
    assert engine.failed_tasks == []

    assert engine.unfinished_tasks == 0
    assert engine.queued_tasks == []
    assert engine.running_tasks == []

    results = engine.results()

    assert len(results) == 1
    assert results[0].expert_id == 43
    assert results[0].status == PrefetchStatus.CANCELLED
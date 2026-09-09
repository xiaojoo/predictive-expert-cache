from __future__ import annotations

import time

from predictive_cache.prefetch.engine import PrefetchEngine
from predictive_cache.prefetch.stages import (
    PrefetchStage,
    PrefetchStageChain,
)
from predictive_cache.prefetch.types import (
    PrefetchSource,
    PrefetchTarget,
    PrefetchTask,
)
from predictive_cache.prefetch.transfer import PrefetchTransferExecutor


def _wait_terminal(
    engine: PrefetchEngine,
    expert_id: int,
    timeout: float = 5.0,
) -> str:
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        status = engine.task_status(expert_id)

        if status in {
            "completed",
            "failed",
            "cancelled",
        }:
            return status

        time.sleep(0.005)

    raise AssertionError(
        f"task {expert_id} did not reach terminal state"
    )


def _make_executor(
    stage_one,
    stage_two,
) -> PrefetchTransferExecutor:
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
            AssertionError("unexpected default handler")
        )
    )

    executor.register(
        PrefetchSource.NVME,
        PrefetchTarget.RAM,
        chain,
    )

    return executor


def test_engine_marks_multistage_stage_one_failure_failed() -> None:
    events: list[str] = []

    def stage_one(task: PrefetchTask) -> None:
        events.append("nvme_to_ram")
        raise RuntimeError("NVMe failure")

    def stage_two(task: PrefetchTask) -> None:
        events.append("ram_to_gpu")

    executor = _make_executor(
        stage_one,
        stage_two,
    )

    engine = PrefetchEngine(
        lambda task: (_ for _ in ()).throw(
            AssertionError("fallback loader must not run")
        ),
        num_workers=1,
        transfer_executor=executor,
    )
    engine.start()

    try:
        task = PrefetchTask(
            expert_id=1001,
            source=PrefetchSource.NVME,
            target=PrefetchTarget.RAM,
        )

        assert engine.submit(task)

        assert _wait_terminal(engine, 1001) == "failed"

        assert events == ["nvme_to_ram"]
        assert len(engine.completed_tasks) == 0
        assert len(engine.failed_tasks) == 1
        assert len(engine.cancelled_tasks) == 0
        assert engine.unfinished_tasks == 0

    finally:
        if engine.running:
            engine.stop(wait=True)


def test_engine_marks_multistage_stage_two_failure_failed() -> None:
    events: list[str] = []

    def stage_one(task: PrefetchTask) -> None:
        events.append("nvme_to_ram")

    def stage_two(task: PrefetchTask) -> None:
        events.append("ram_to_gpu")
        raise RuntimeError("GPU failure")

    executor = _make_executor(
        stage_one,
        stage_two,
    )

    engine = PrefetchEngine(
        lambda task: (_ for _ in ()).throw(
            AssertionError("fallback loader must not run")
        ),
        num_workers=1,
        transfer_executor=executor,
    )
    engine.start()

    try:
        task = PrefetchTask(
            expert_id=1002,
            source=PrefetchSource.NVME,
            target=PrefetchTarget.RAM,
        )

        assert engine.submit(task)

        assert _wait_terminal(engine, 1002) == "failed"

        assert events == [
            "nvme_to_ram",
            "ram_to_gpu",
        ]

        assert len(engine.completed_tasks) == 0
        assert len(engine.failed_tasks) == 1
        assert len(engine.cancelled_tasks) == 0
        assert engine.unfinished_tasks == 0

    finally:
        if engine.running:
            engine.stop(wait=True)
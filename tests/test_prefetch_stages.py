from __future__ import annotations

from predictive_cache.prefetch.stages import (
    PrefetchStage,
    PrefetchStageChain,
)
from predictive_cache.prefetch.types import (
    PrefetchSource,
    PrefetchTarget,
    PrefetchTask,
)


def test_prefetch_stage_chain_executes_in_order() -> None:
    events: list[tuple[int, object, object]] = []

    def stage_one(task: PrefetchTask) -> None:
        events.append(
            (
                int(task.expert_id),
                task.source,
                task.target,
            )
        )

    def stage_two(task: PrefetchTask) -> None:
        events.append(
            (
                int(task.expert_id),
                task.source,
                task.target,
            )
        )

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

    chain.execute(
        PrefetchTask(
            expert_id=26,
            source=PrefetchSource.NVME,
            target=PrefetchTarget.RAM,
        )
    )

    assert events == [
        (
            26,
            PrefetchSource.NVME,
            PrefetchTarget.RAM,
        ),
        (
            26,
            PrefetchSource.RAM,
            PrefetchTarget.GPU,
        ),
    ]

def test_prefetch_stage_chain_stops_after_failure() -> None:
    events: list[str] = []

    def stage_one(task: PrefetchTask) -> None:
        events.append("stage1")
        raise RuntimeError("stage1 failed")

    def stage_two(task: PrefetchTask) -> None:
        events.append("stage2")

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

    try:
        chain.execute(
            PrefetchTask(
                expert_id=26,
                source=PrefetchSource.NVME,
                target=PrefetchTarget.RAM,
            )
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected stage failure")

    assert events == ["stage1"]

def test_prefetch_stage_chain_preserves_task_metadata() -> None:
    observed: list[PrefetchTask] = []

    def stage_one(task: PrefetchTask) -> None:
        observed.append(task)

    def stage_two(task: PrefetchTask) -> None:
        observed.append(task)

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

    task = PrefetchTask(
        expert_id=26,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
        priority=0.75,
        confidence=0.9,
        estimated_distance=3,
    )

    chain.execute(task)

    assert len(observed) == 2

    assert observed[0].expert_id == 26
    assert observed[1].expert_id == 26

    assert observed[0].priority == 0.75
    assert observed[1].priority == 0.75

    assert observed[0].confidence == 0.9
    assert observed[1].confidence == 0.9

    assert observed[0].estimated_distance == 3
    assert observed[1].estimated_distance == 3

    assert observed[0].source == PrefetchSource.NVME
    assert observed[0].target == PrefetchTarget.RAM

    assert observed[1].source == PrefetchSource.RAM
    assert observed[1].target == PrefetchTarget.GPU


def test_prefetch_stage_chain_reports_final_target() -> None:
    chain = PrefetchStageChain(
        [
            PrefetchStage(
                source=PrefetchSource.NVME,
                target=PrefetchTarget.RAM,
                handler=lambda task: None,
            ),
            PrefetchStage(
                source=PrefetchSource.RAM,
                target=PrefetchTarget.GPU,
                handler=lambda task: None,
            ),
        ]
    )

    assert chain.final_target == PrefetchTarget.GPU

def test_prefetch_stage_chain_stage_one_failure_skips_stage_two() -> None:
    events: list[str] = []

    def stage_one(task: PrefetchTask) -> None:
        events.append("nvme_to_ram")
        raise RuntimeError("NVMe read failed")

    def stage_two(task: PrefetchTask) -> None:
        events.append("ram_to_gpu")

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

    try:
        chain.execute(
            PrefetchTask(
                expert_id=26,
                source=PrefetchSource.NVME,
                target=PrefetchTarget.RAM,
            )
        )
    except RuntimeError as exc:
        assert str(exc) == "NVMe read failed"
    else:
        raise AssertionError("expected stage-one failure")

    assert events == ["nvme_to_ram"]


def test_prefetch_stage_chain_stage_two_failure_propagates() -> None:
    events: list[str] = []

    def stage_one(task: PrefetchTask) -> None:
        events.append("nvme_to_ram")

    def stage_two(task: PrefetchTask) -> None:
        events.append("ram_to_gpu")
        raise RuntimeError("GPU transfer failed")

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

    try:
        chain.execute(
            PrefetchTask(
                expert_id=26,
                source=PrefetchSource.NVME,
                target=PrefetchTarget.RAM,
            )
        )
    except RuntimeError as exc:
        assert str(exc) == "GPU transfer failed"
    else:
        raise AssertionError("expected stage-two failure")

    assert events == [
        "nvme_to_ram",
        "ram_to_gpu",
    ]
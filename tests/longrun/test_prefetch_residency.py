from __future__ import annotations

import time

from predictive_cache.prefetch import (
    PrefetchEngine,
    PrefetchSource,
    PrefetchStatus,
    PrefetchTarget,
    PrefetchTask,
)
from predictive_cache.prefetch.storage import (
    create_storage_transfer_executor,
)
from predictive_cache.storage.expert_record import ExpertRecord
from predictive_cache.storage.expert_store import ExpertLocation
from predictive_cache.storage.in_memory import InMemoryExpertStore


def _make_task(expert_id: int) -> PrefetchTask:
    return PrefetchTask(
        expert_id=expert_id,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
        priority=1.0,
        confidence=0.9,
    )


def _wait_for_status(
    engine: PrefetchEngine,
    expert_id: int,
    status: PrefetchStatus,
    timeout: float = 2.0,
) -> bool:
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if engine.task_status(expert_id) == status:
            return True
        time.sleep(0.001)

    return False


def _run_residency_trace(
    cycles: int = 1000,
) -> list[tuple[int, object]]:
    expert_count = 16

    nvme = InMemoryExpertStore()
    ram = InMemoryExpertStore()

    for expert_id in range(expert_count):
        nvme.put(
            ExpertRecord(
                expert_id=expert_id,
                location=ExpertLocation.NVME,
                payload=f"expert-{expert_id}",
            )
        )

    fallback_calls: list[int] = []

    def fallback(task: PrefetchTask) -> None:
        fallback_calls.append(task.expert_id)

    transfer = create_storage_transfer_executor(
        nvme,
        ram,
        fallback,
    )

    engine = PrefetchEngine(
        fallback,
        num_workers=1,
        transfer_executor=transfer,
    )

    engine.start()

    snapshots: list[tuple[int, object]] = []

    try:
        for cycle in range(cycles):
            expert_id = cycle % expert_count
            task = _make_task(expert_id)

            assert engine.submit(task) is True

            assert _wait_for_status(
                engine,
                expert_id,
                PrefetchStatus.COMPLETED,
            )

            record = ram.get(expert_id)

            assert record is not None
            assert record.location == ExpertLocation.RAM
            assert record.payload == f"expert-{expert_id}"

            snapshots.append(
                (
                    expert_id,
                    record.payload,
                )
            )

        assert fallback_calls == []
        assert len(snapshots) == cycles
        assert all(
            ram.contains(expert_id)
            for expert_id in range(expert_count)
        )

    finally:
        engine.stop()

    return snapshots


def test_real_nvme_to_ram_residency_1000_cycles() -> None:
    snapshots = _run_residency_trace(1000)

    assert len(snapshots) == 1000
    assert snapshots[0] == (0, "expert-0")
    assert snapshots[-1] == (7, "expert-7")


def test_real_nvme_to_ram_residency_is_deterministic() -> None:
    snapshots_a = _run_residency_trace(1000)
    snapshots_b = _run_residency_trace(1000)

    assert snapshots_a == snapshots_b
from __future__ import annotations

import tempfile
import time
from pathlib import Path
from threading import Event

from predictive_cache.prefetch.engine import PrefetchEngine
from predictive_cache.prefetch.stages import (
    PrefetchStage,
    PrefetchStageChain,
)
from predictive_cache.prefetch.storage import (
    create_storage_transfer_executor,
)
from predictive_cache.prefetch.types import (
    PrefetchSource,
    PrefetchTarget,
    PrefetchTask,
)
from predictive_cache.storage.expert_record import ExpertRecord
from predictive_cache.storage.expert_store import ExpertLocation
from predictive_cache.storage.gpu import GPUExpertStore
from predictive_cache.storage.nvme import NVMeExpertStore
from predictive_cache.storage.ram import RAMExpertStore


def _wait_terminal(
    engine: PrefetchEngine,
    expert_id: int,
    timeout: float = 10.0,
) -> str:
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        status = engine.task_status(expert_id)

        if status in {"completed", "failed", "cancelled"}:
            return status

        time.sleep(0.005)

    raise AssertionError(
        f"task {expert_id} did not reach terminal state"
    )


def _task(expert_id: int) -> PrefetchTask:
    return PrefetchTask(
        expert_id=expert_id,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
    )


def _assert_idle(engine: PrefetchEngine) -> None:
    assert engine.unfinished_tasks == 0
    assert engine.queue_size == 0
    assert len(engine.queued_tasks) == 0
    assert len(engine.running_tasks) == 0


def test_r4_stage_one_failure_reclaims_task_state() -> None:
    expert_id = 11
    payload = b"stage-one-failure" * 128

    with tempfile.TemporaryDirectory() as tmp:
        nvme = NVMeExpertStore(root_dir=Path(tmp) / "nvme")
        ram = RAMExpertStore()
        gpu = GPUExpertStore()

        nvme.put(
            ExpertRecord(
                expert_id=expert_id,
                location=ExpertLocation.NVME,
                payload=payload,
            )
        )

        attempts = {"count": 0}

        real_transfer = create_storage_transfer_executor(
            nvme,
            ram,
            lambda task: (_ for _ in ()).throw(
                AssertionError("fallback loader must not run")
            ),
            gpu=gpu,
        )

        def failing_stage_one(task: PrefetchTask) -> None:
            attempts["count"] += 1
            raise RuntimeError("injected stage-one failure")

        chain = PrefetchStageChain(
            [
                PrefetchStage(
                    source=PrefetchSource.NVME,
                    target=PrefetchTarget.RAM,
                    handler=failing_stage_one,
                ),
            ]
        )

        from predictive_cache.prefetch.transfer import PrefetchTransferExecutor

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
            assert engine.submit(_task(expert_id))

            assert _wait_terminal(engine, expert_id) == "failed"

            assert attempts["count"] == 1
            assert len(engine.completed_tasks) == 0
            assert len(engine.failed_tasks) == 1
            assert len(engine.cancelled_tasks) == 0

            assert ram.get(expert_id) is None
            assert gpu.get(expert_id) is None

            _assert_idle(engine)

        finally:
            if engine.running:
                engine.stop(wait=True)


def test_r4_stage_two_failure_preserves_stage_one_residency() -> None:
    expert_id = 12
    payload = b"stage-two-failure" * 128

    with tempfile.TemporaryDirectory() as tmp:
        nvme = NVMeExpertStore(root_dir=Path(tmp) / "nvme")
        ram = RAMExpertStore()
        gpu = GPUExpertStore()

        nvme.put(
            ExpertRecord(
                expert_id=expert_id,
                location=ExpertLocation.NVME,
                payload=payload,
            )
        )

        real_nvme_to_ram = (
            __import__(
                "predictive_cache.prefetch.storage",
                fromlist=["create_nvme_to_ram_handler"],
            )
            .create_nvme_to_ram_handler(nvme, ram)
        )

        stage_two_attempts = {"count": 0}

        def failing_stage_two(task: PrefetchTask) -> None:
            stage_two_attempts["count"] += 1
            raise RuntimeError("injected stage-two failure")

        chain = PrefetchStageChain(
            [
                PrefetchStage(
                    source=PrefetchSource.NVME,
                    target=PrefetchTarget.RAM,
                    handler=real_nvme_to_ram,
                ),
                PrefetchStage(
                    source=PrefetchSource.RAM,
                    target=PrefetchTarget.GPU,
                    handler=failing_stage_two,
                ),
            ]
        )

        from predictive_cache.prefetch.transfer import PrefetchTransferExecutor

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
            assert engine.submit(_task(expert_id))

            assert _wait_terminal(engine, expert_id) == "failed"

            assert stage_two_attempts["count"] == 1
            assert len(engine.completed_tasks) == 0
            assert len(engine.failed_tasks) == 1
            assert len(engine.cancelled_tasks) == 0

            ram_record = ram.get(expert_id)
            assert ram_record is not None
            assert ram_record.payload == payload

            assert gpu.get(expert_id) is None

            _assert_idle(engine)

        finally:
            if engine.running:
                engine.stop(wait=True)


def test_r4_failed_stage_two_can_retry_to_completion() -> None:
    expert_id = 13
    payload = b"retry-after-stage-two-failure" * 128

    with tempfile.TemporaryDirectory() as tmp:
        nvme = NVMeExpertStore(root_dir=Path(tmp) / "nvme")
        ram = RAMExpertStore()
        gpu = GPUExpertStore()

        nvme.put(
            ExpertRecord(
                expert_id=expert_id,
                location=ExpertLocation.NVME,
                payload=payload,
            )
        )

        from predictive_cache.prefetch.storage import (
            create_nvme_to_ram_handler,
            create_ram_to_gpu_handler,
        )
        from predictive_cache.prefetch.transfer import (
            PrefetchTransferExecutor,
        )

        real_stage_one = create_nvme_to_ram_handler(nvme, ram)
        real_stage_two = create_ram_to_gpu_handler(ram, gpu)

        attempts = {"stage_two": 0}

        def retryable_stage_two(task: PrefetchTask) -> None:
            attempts["stage_two"] += 1

            if attempts["stage_two"] == 1:
                raise RuntimeError("injected first stage-two failure")

            real_stage_two(task)

        chain = PrefetchStageChain(
            [
                PrefetchStage(
                    source=PrefetchSource.NVME,
                    target=PrefetchTarget.RAM,
                    handler=real_stage_one,
                ),
                PrefetchStage(
                    source=PrefetchSource.RAM,
                    target=PrefetchTarget.GPU,
                    handler=retryable_stage_two,
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
            assert engine.submit(_task(expert_id))

            assert _wait_terminal(engine, expert_id) == "failed"

            assert len(engine.failed_tasks) == 1
            assert len(engine.completed_tasks) == 0
            assert len(engine.cancelled_tasks) == 0
            _assert_idle(engine)

            assert ram.contains(expert_id)
            assert gpu.get(expert_id) is None

            assert engine.submit(_task(expert_id))

            assert _wait_terminal(engine, expert_id) == "completed"

            assert attempts["stage_two"] == 2

            assert len(engine.completed_tasks) == 1
            assert len(engine.failed_tasks) == 0
            assert len(engine.cancelled_tasks) == 0

            ram_record = ram.get(expert_id)
            assert ram_record is not None
            assert ram_record.payload == payload

            gpu_record = gpu.get(expert_id)
            assert gpu_record is not None
            assert gpu_record.payload == payload

            _assert_idle(engine)

        finally:
            if engine.running:
                engine.stop(wait=True)


def test_r4_cancelled_task_reclaims_queue_and_running_state() -> None:
    expert_id = 14
    payload = b"cancel-reclamation" * 128

    with tempfile.TemporaryDirectory() as tmp:
        nvme = NVMeExpertStore(root_dir=Path(tmp) / "nvme")
        ram = RAMExpertStore()
        gpu = GPUExpertStore()

        nvme.put(
            ExpertRecord(
                expert_id=expert_id,
                location=ExpertLocation.NVME,
                payload=payload,
            )
        )

        cancellation = Event()

        from predictive_cache.prefetch.storage import (
            create_nvme_to_ram_handler,
            create_ram_to_gpu_handler,
        )
        from predictive_cache.prefetch.transfer import (
            PrefetchTransferExecutor,
        )

        stage_one = create_nvme_to_ram_handler(nvme, ram)
        stage_two = create_ram_to_gpu_handler(ram, gpu)

        def tracked_stage_one(task: PrefetchTask) -> None:
            stage_one(task)
            cancellation.set()

        chain = PrefetchStageChain(
            [
                PrefetchStage(
                    source=PrefetchSource.NVME,
                    target=PrefetchTarget.RAM,
                    handler=tracked_stage_one,
                ),
                PrefetchStage(
                    source=PrefetchSource.RAM,
                    target=PrefetchTarget.GPU,
                    handler=stage_two,
                ),
            ],
            should_cancel=cancellation.is_set,
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
            assert engine.submit(_task(expert_id))

            assert _wait_terminal(engine, expert_id) == "cancelled"

            assert len(engine.completed_tasks) == 0
            assert len(engine.failed_tasks) == 0
            assert len(engine.cancelled_tasks) == 1

            assert ram.contains(expert_id)
            assert gpu.get(expert_id) is None

            _assert_idle(engine)

        finally:
            if engine.running:
                engine.stop(wait=True)

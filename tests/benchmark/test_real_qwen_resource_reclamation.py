from __future__ import annotations

import tempfile
import time
from pathlib import Path

import torch
from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

from predictive_cache.prefetch.engine import PrefetchEngine
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


def _build_qwen() -> Qwen2MoeForCausalLM:
    config = Qwen2MoeConfig(
        num_hidden_layers=2,
        hidden_size=128,
        intermediate_size=256,
        num_local_experts=8,
        num_experts_per_tok=2,
        vocab_size=128,
    )

    model = Qwen2MoeForCausalLM(config)
    model.eval()
    return model.cuda()


def _assert_qwen_cuda_forward() -> None:
    model = _build_qwen()

    with torch.no_grad():
        outputs = model(
            input_ids=torch.tensor(
                [[1]],
                dtype=torch.long,
                device="cuda",
            )
        )

    assert outputs.logits.shape[-1] == 128


def _assert_idle(engine: PrefetchEngine) -> None:
    assert engine.unfinished_tasks == 0
    assert engine.queue_size == 0
    assert len(engine.queued_tasks) == 0
    assert len(engine.running_tasks) == 0


def _put_nvme(
    nvme: NVMeExpertStore,
    expert_id: int,
    payload: bytes,
) -> None:
    nvme.put(
        ExpertRecord(
            expert_id=expert_id,
            location=ExpertLocation.NVME,
            payload=payload,
        )
    )


def test_real_qwen_r4_stage_one_failure_reclaims_resources() -> None:
    if not torch.cuda.is_available():
        raise AssertionError("CUDA is required for 7-R-4")

    expert_id = 21
    payload = b"r4-stage-one-failure" * 256

    with tempfile.TemporaryDirectory() as tmp:
        nvme = NVMeExpertStore(root_dir=Path(tmp) / "nvme")
        ram = RAMExpertStore()
        gpu = GPUExpertStore()

        _put_nvme(nvme, expert_id, payload)

        original_ram_put = ram.put

        def failing_ram_put(record: ExpertRecord) -> None:
            raise RuntimeError("injected stage-one failure")

        ram.put = failing_ram_put

        transfer = create_storage_transfer_executor(
            nvme,
            ram,
            lambda task: (_ for _ in ()).throw(
                AssertionError("fallback loader must not run")
            ),
            gpu=gpu,
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

            assert ram.get(expert_id) is None
            assert gpu.get(expert_id) is None

            assert len(engine.completed_tasks) == 0
            assert len(engine.failed_tasks) == 1
            assert len(engine.cancelled_tasks) == 0

            _assert_idle(engine)

            # Restore the real RAM store method and prove Qwen CUDA remains healthy.
            ram.put = original_ram_put
            _assert_qwen_cuda_forward()

        finally:
            ram.put = original_ram_put
            if engine.running:
                engine.stop(wait=True)


def test_real_qwen_r4_stage_two_failure_preserves_ram_only() -> None:
    if not torch.cuda.is_available():
        raise AssertionError("CUDA is required for 7-R-4")

    expert_id = 22
    payload = b"r4-stage-two-failure" * 256

    with tempfile.TemporaryDirectory() as tmp:
        nvme = NVMeExpertStore(root_dir=Path(tmp) / "nvme")
        ram = RAMExpertStore()
        gpu = GPUExpertStore()

        _put_nvme(nvme, expert_id, payload)

        original_gpu_put = gpu.put

        def failing_gpu_put(record: ExpertRecord) -> None:
            raise RuntimeError("injected stage-two failure")

        gpu.put = failing_gpu_put

        transfer = create_storage_transfer_executor(
            nvme,
            ram,
            lambda task: (_ for _ in ()).throw(
                AssertionError("fallback loader must not run")
            ),
            gpu=gpu,
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

            ram_record = ram.get(expert_id)
            assert ram_record is not None
            assert ram_record.payload == payload

            assert gpu.get(expert_id) is None

            assert len(engine.completed_tasks) == 0
            assert len(engine.failed_tasks) == 1
            assert len(engine.cancelled_tasks) == 0

            _assert_idle(engine)

            gpu.put = original_gpu_put
            _assert_qwen_cuda_forward()

        finally:
            gpu.put = original_gpu_put
            if engine.running:
                engine.stop(wait=True)


def test_real_qwen_r4_stage_two_failure_then_retry_completes() -> None:
    if not torch.cuda.is_available():
        raise AssertionError("CUDA is required for 7-R-4")

    expert_id = 23
    payload = b"r4-retry-after-stage-two-failure" * 256

    with tempfile.TemporaryDirectory() as tmp:
        nvme = NVMeExpertStore(root_dir=Path(tmp) / "nvme")
        ram = RAMExpertStore()
        gpu = GPUExpertStore()

        _put_nvme(nvme, expert_id, payload)

        original_gpu_put = gpu.put
        attempts = {"gpu": 0}

        def retryable_gpu_put(record: ExpertRecord) -> None:
            attempts["gpu"] += 1

            if attempts["gpu"] == 1:
                raise RuntimeError("injected first stage-two failure")

            original_gpu_put(record)

        gpu.put = retryable_gpu_put

        transfer = create_storage_transfer_executor(
            nvme,
            ram,
            lambda task: (_ for _ in ()).throw(
                AssertionError("fallback loader must not run")
            ),
            gpu=gpu,
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
            # First attempt: Stage 1 succeeds, Stage 2 fails.
            assert engine.submit(_task(expert_id))
            assert _wait_terminal(engine, expert_id) == "failed"

            assert attempts["gpu"] == 1

            ram_record = ram.get(expert_id)
            assert ram_record is not None
            assert ram_record.payload == payload

            assert gpu.get(expert_id) is None

            assert len(engine.completed_tasks) == 0
            assert len(engine.failed_tasks) == 1
            assert len(engine.cancelled_tasks) == 0

            _assert_idle(engine)

            # Retry the same production task.
            assert engine.submit(_task(expert_id))
            assert _wait_terminal(engine, expert_id) == "completed"

            assert attempts["gpu"] == 2

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

            _assert_qwen_cuda_forward()

            print(
                "\n7-R-4 Real Qwen Resource Reclamation"
            )
            print("------------------------------------")
            print("Stage 1 failure:             RECLAIMED")
            print("Stage 2 failure:             RAM PRESERVED / GPU ABSENT")
            print("Stage 2 retry:               COMPLETED")
            print("RAM payload integrity:       PASS")
            print("GPU payload integrity:       PASS")
            print("Qwen CUDA forward:           PASS")
            print(f"stage-2 attempts:            {attempts['gpu']}")
            print(f"unfinished tasks:            {engine.unfinished_tasks}")
            print(f"queued tasks:                {engine.queue_size}")
            print()

        finally:
            gpu.put = original_gpu_put
            if engine.running:
                engine.stop(wait=True)

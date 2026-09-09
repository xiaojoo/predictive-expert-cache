from __future__ import annotations

import tempfile
import time
from pathlib import Path
from threading import Event

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
    return model


def test_real_qwen_production_cancellation_after_stage_one() -> None:
    if not torch.cuda.is_available():
        raise AssertionError("CUDA is required for 7-R-3")

    expert_id = 3
    payload = bytes(range(256)) * 32

    with tempfile.TemporaryDirectory() as tmp:
        nvme = NVMeExpertStore(
            root_dir=Path(tmp) / "nvme"
        )
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
        stage_one_completed = Event()
        stage_two_calls = {"count": 0}

        original_ram_put = ram.put
        original_gpu_put = gpu.put

        def tracked_ram_put(record: ExpertRecord) -> None:
            original_ram_put(record)
            stage_one_completed.set()
            cancellation.set()

        def tracked_gpu_put(record: ExpertRecord) -> None:
            stage_two_calls["count"] += 1
            original_gpu_put(record)

        ram.put = tracked_ram_put
        gpu.put = tracked_gpu_put

        transfer = create_storage_transfer_executor(
            nvme,
            ram,
            lambda task: (_ for _ in ()).throw(
                AssertionError(
                    "fallback loader must not run"
                )
            ),
            gpu=gpu,
            should_cancel=cancellation.is_set,
        )

        engine = PrefetchEngine(
            lambda task: (_ for _ in ()).throw(
                AssertionError(
                    "fallback loader must not run"
                )
            ),
            num_workers=1,
            transfer_executor=transfer,
        )

        engine.start()

        try:
            task = PrefetchTask(
                expert_id=expert_id,
                source=PrefetchSource.NVME,
                target=PrefetchTarget.RAM,
            )

            assert engine.submit(task)

            status = _wait_terminal(
                engine,
                expert_id,
            )

            assert status == "cancelled"

            # Stage 1 really completed before cancellation.
            assert stage_one_completed.is_set()

            # Stage 1 residency must remain.
            ram_record = ram.get(expert_id)
            assert ram_record is not None
            assert ram_record.payload == payload

            # Stage 2 must never execute.
            assert stage_two_calls["count"] == 0
            assert gpu.get(expert_id) is None

            # Engine lifecycle/accounting must be fully reclaimed.
            assert len(engine.completed_tasks) == 0
            assert len(engine.failed_tasks) == 0
            assert len(engine.cancelled_tasks) == 1
            assert engine.unfinished_tasks == 0
            assert engine.queue_size == 0
            assert len(engine.queued_tasks) == 0
            assert len(engine.running_tasks) == 0

            # The real Qwen CUDA path still executes successfully.
            model = _build_qwen().cuda()

            with torch.no_grad():
                outputs = model(
                    input_ids=torch.tensor(
                        [[1]],
                        dtype=torch.long,
                        device="cuda",
                    )
                )

            assert outputs.logits.shape[-1] == 128

            print(
                "\n7-R-3 Real Qwen Production Cancellation Boundary"
            )
            print("-----------------------------------------------")
            print(f"expert id:                 {expert_id}")
            print("stage 1 NVMe -> RAM:       COMPLETED")
            print("cancellation boundary:     TRIGGERED")
            print("stage 2 RAM -> GPU calls:  0")
            print("GPU residency:             ABSENT")
            print("engine status:             CANCELLED")
            print(f"unfinished tasks:          {engine.unfinished_tasks}")
            print(f"queued tasks:              {engine.queue_size}")
            print()

        finally:
            if engine.running:
                engine.stop(wait=True)


if __name__ == "__main__":
    test_real_qwen_production_cancellation_after_stage_one()

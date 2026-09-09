from __future__ import annotations

import tempfile
import time
from pathlib import Path

import torch
from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

from predictive_cache.prefetch.engine import PrefetchEngine
from predictive_cache.prefetch.stages import (
    PrefetchStage,
    PrefetchStageChain,
)
from predictive_cache.prefetch.storage import (
    create_nvme_to_ram_handler,
    create_ram_to_gpu_handler,
)
from predictive_cache.prefetch.transfer import PrefetchTransferExecutor
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
    expert_id: str,
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


def test_real_qwen_multistage_failure_then_retry_completes() -> None:
    if not torch.cuda.is_available():
        raise AssertionError("CUDA is required for 7-Q-3")

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

        real_nvme_to_ram = create_nvme_to_ram_handler(
            nvme,
            ram,
        )

        real_ram_to_gpu = create_ram_to_gpu_handler(
            ram,
            gpu,
        )

        attempts = {"stage_one": 0}

        def flaky_stage_one(task: PrefetchTask) -> None:
            attempts["stage_one"] += 1

            if attempts["stage_one"] == 1:
                raise RuntimeError("injected NVMe failure")

            real_nvme_to_ram(task)

        chain = PrefetchStageChain(
            [
                PrefetchStage(
                    source=PrefetchSource.NVME,
                    target=PrefetchTarget.RAM,
                    handler=flaky_stage_one,
                ),
                PrefetchStage(
                    source=PrefetchSource.RAM,
                    target=PrefetchTarget.GPU,
                    handler=real_ram_to_gpu,
                ),
            ]
        )

        transfer = PrefetchTransferExecutor(
            lambda task: (_ for _ in ()).throw(
                AssertionError(
                    "fallback loader must not run"
                )
            )
        )

        transfer.register(
            PrefetchSource.NVME,
            PrefetchTarget.RAM,
            chain,
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
            first = PrefetchTask(
                expert_id=expert_id,
                source=PrefetchSource.NVME,
                target=PrefetchTarget.RAM,
            )

            assert engine.submit(first)

            assert _wait_terminal(
                engine,
                expert_id,
            ) == "failed"

            assert len(engine.completed_tasks) == 0
            assert len(engine.failed_tasks) == 1
            assert len(engine.cancelled_tasks) == 0
            assert engine.unfinished_tasks == 0

            assert not ram.contains(expert_id)
            assert gpu.get(expert_id) is None

            second = PrefetchTask(
                expert_id=expert_id,
                source=PrefetchSource.NVME,
                target=PrefetchTarget.RAM,
            )

            assert engine.submit(second)

            assert _wait_terminal(
                engine,
                expert_id,
            ) == "completed"

            assert attempts["stage_one"] == 2

            assert len(engine.completed_tasks) == 1
            assert len(engine.failed_tasks) == 0
            assert len(engine.cancelled_tasks) == 0
            assert engine.unfinished_tasks == 0

            assert ram.contains(expert_id)

            gpu_record = gpu.get(expert_id)
            assert gpu_record is not None
            assert gpu_record.payload == payload

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

        finally:
            if engine.running:
                engine.stop(wait=True)
from __future__ import annotations

import shutil
from pathlib import Path
from threading import Event

import pytest
import torch
from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

from predictive_cache import CacheConfig, PredictiveExpertCache
from predictive_cache.prefetch.engine import PrefetchEngine
from predictive_cache.prefetch.pipeline import PrefetchPipeline
from predictive_cache.prefetch.stages import PrefetchCancelled
from predictive_cache.prefetch.storage import (
    create_storage_transfer_executor,
)
from predictive_cache.prefetch.types import (
    PrefetchSource,
    PrefetchStatus,
    PrefetchTarget,
    PrefetchTask,
)
from predictive_cache.routing import (
    QwenMoeRoutingCapture,
    QwenRoutingBridge,
)
from predictive_cache.scheduler import ExpertScheduler
from predictive_cache.storage.expert_record import ExpertRecord
from predictive_cache.storage.expert_store import (
    ExpertLocation,
    ExpertStore,
)
from predictive_cache.storage.gpu import GPUExpertStore
from predictive_cache.storage.nvme import NVMeExpertStore
from predictive_cache.storage.ram import RAMExpertStore


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="7-R-3 requires CUDA",
)


def _make_model() -> Qwen2MoeForCausalLM:
    return Qwen2MoeForCausalLM(
        Qwen2MoeConfig(
            num_hidden_layers=2,
            hidden_size=128,
            intermediate_size=256,
            num_local_experts=8,
            num_experts_per_tok=2,
            vocab_size=128,
        )
    ).eval().cuda()


def _capture_current(
    capture: QwenMoeRoutingCapture,
    model: Qwen2MoeForCausalLM,
    input_ids: torch.Tensor,
    step: int,
) -> list[int]:
    before = len(capture.bridge.collector.events())

    with torch.no_grad():
        capture.capture_forward(
            model,
            input_ids=input_ids.cuda(),
            step=step,
        )

    events = capture.bridge.collector.events()[before:]

    current: list[int] = []

    for event in events:
        if event.step != step:
            continue

        current.extend(
            int(x)
            for x in event.expert_ids
        )

    return sorted(set(current))


def _wait_terminal(
    engine: PrefetchEngine,
    expert_ids: list[int],
    timeout_s: float = 10.0,
) -> None:
    import time

    deadline = time.perf_counter() + timeout_s

    while time.perf_counter() < deadline:
        if all(
            engine.task_status(expert_id)
            in {
                PrefetchStatus.COMPLETED,
                PrefetchStatus.FAILED,
                PrefetchStatus.CANCELLED,
            }
            for expert_id in expert_ids
        ):
            return

        time.sleep(0.001)

    raise AssertionError(
        f"tasks did not reach terminal state: {expert_ids}"
    )


def test_real_qwen_multistage_cancellation_boundary() -> None:
    torch.manual_seed(7)

    warmup_steps = 32

    root = Path(
        ".pytest_nvme_qwen_multistage_cancellation"
    )

    if root.exists():
        shutil.rmtree(root)

    root.mkdir(parents=True)

    model = _make_model()

    cache = PredictiveExpertCache(
        CacheConfig(
            capacity=8,
            prediction_top_k=3,
        )
    )

    scheduler = ExpertScheduler(cache)

    nvme = NVMeExpertStore(root)
    ram = RAMExpertStore()
    gpu = GPUExpertStore("cuda")

    bridge = QwenRoutingBridge(cache)
    capture = QwenMoeRoutingCapture(bridge)

    payload = torch.randn(
        2048,
        1024,
        dtype=torch.float32,
    ).contiguous()

    source_payloads: dict[int, torch.Tensor] = {}

    stage1_calls: list[int] = []
    stage2_calls: list[int] = []

    cancelled = Event()
    target_expert: int | None = None

    def should_cancel() -> bool:
        return cancelled.is_set()

    def fallback_loader(task: PrefetchTask) -> None:
        raise AssertionError(
            "fallback loader was invoked; "
            "production transfer executor did not route "
            "the task through NVMe -> RAM -> GPU"
        )

    transfer = create_storage_transfer_executor(
        nvme,
        ram,
        fallback_loader,
        gpu=gpu,
        should_cancel=should_cancel,
    )

    engine = PrefetchEngine(
        loader=fallback_loader,
        max_queue_size=8,
        transfer_executor=transfer,
    )

    pipeline = PrefetchPipeline(
        scheduler,
        engine,
        auto_start=True,
    )

    try:
        # ----------------------------------------------------------
        # Warm-up: establish real Qwen prediction history.
        # ----------------------------------------------------------
        for step in range(warmup_steps):
            input_ids = torch.tensor(
                [[10 + (step % 118)]],
                dtype=torch.long,
                device="cuda",
            )

            current = _capture_current(
                capture,
                model,
                input_ids,
                step,
            )

            if current:
                scheduler.plan_prefetch_predictions(
                    current
                )

        # ----------------------------------------------------------
        # Find one real predicted expert.
        # ----------------------------------------------------------
        current: list[int] = []

        for step in range(warmup_steps, warmup_steps + 8):
            input_ids = torch.tensor(
                [[10 + (step % 118)]],
                dtype=torch.long,
                device="cuda",
            )

            current = _capture_current(
                capture,
                model,
                input_ids,
                step,
            )

            if current:
                candidates = (
                    scheduler.plan_prefetch_predictions(
                        current
                    )
                )

                if candidates:
                    target_expert = int(
                        candidates[0].expert_id
                    )
                    break

        assert target_expert is not None

        expert_id = target_expert

        # ----------------------------------------------------------
        # Materialize exactly the selected expert on real NVMe.
        # ----------------------------------------------------------
        source_payloads[expert_id] = payload.clone()

        nvme.put(
            ExpertRecord(
                expert_id=expert_id,
                location=ExpertLocation.NVME,
                payload=payload.clone(),
            )
        )

        # ----------------------------------------------------------
        # Instrument production transfer handlers.
        #
        # The cancellation checkpoint itself remains inside the
        # production PrefetchStageChain. These counters only verify
        # physical residency effects.
        # ----------------------------------------------------------
        # ----------------------------------------------------------
        # First-stage cancellation boundary.
        #
        # The production NVMe -> RAM handler executes first.
        # After Stage 1 returns, this callback becomes true.
        #
        # To create that exact boundary deterministically, the
        # Stage 1 handler must signal completion before the next
        # stage checkpoint.
        # ----------------------------------------------------------
        stage1_completed = Event()

        def cancellation_check() -> bool:
            return stage1_completed.is_set()

        transfer = create_storage_transfer_executor(
            nvme,
            ram,
            fallback_loader,
            gpu=gpu,
            should_cancel=cancellation_check,
        )

        original_ram_put = ram.put

        def boundary_ram_put(record: ExpertRecord) -> None:
            original_ram_put(record)
            stage1_calls.append(
                int(record.expert_id)
            )

            if int(record.expert_id) == expert_id:
                stage1_completed.set()

        ram.put = boundary_ram_put  # type: ignore[method-assign]

        engine = PrefetchEngine(
            loader=fallback_loader,
            max_queue_size=8,
            transfer_executor=transfer,
        )

        pipeline.stop()

        pipeline = PrefetchPipeline(
            scheduler,
            engine,
            auto_start=True,
        )

        # ----------------------------------------------------------
        # Submit through the real production pipeline.
        # ----------------------------------------------------------
        task = PrefetchTask(
            expert_id=expert_id,
            source=PrefetchSource.NVME,
            target=PrefetchTarget.RAM,
            priority=candidates[0].priority,
            confidence=candidates[0].confidence,
        )

        assert engine.submit(task)

        _wait_terminal(
            engine,
            [expert_id],
        )

        torch.cuda.synchronize()

        # ----------------------------------------------------------
        # R-3 invariants.
        # ----------------------------------------------------------
        assert stage1_calls == [expert_id]
        assert stage2_calls == []

        assert engine.task_status(
            expert_id
        ) == PrefetchStatus.CANCELLED

        assert engine.completed_tasks == []
        assert engine.failed_tasks == []
        assert engine.cancelled_tasks == [
            expert_id
        ]

        assert engine.unfinished_tasks == 0
        assert engine.queued_tasks == []
        assert engine.running_tasks == []

        # Stage 1 has legitimately completed.
        assert ram.contains(expert_id)

        ram_record = ram.get(expert_id)

        assert ram_record is not None
        assert isinstance(
            ram_record.payload,
            torch.Tensor,
        )

        assert ram_record.payload.device.type == "cpu"

        assert torch.equal(
            ram_record.payload,
            source_payloads[expert_id],
        )

        # Stage 2 must NOT have created GPU residency.
        assert not gpu.contains(expert_id)

        results = engine.results()

        assert len(results) == 1
        assert results[0].expert_id == expert_id
        assert results[0].status == PrefetchStatus.CANCELLED

        print()
        print(
            "7-R-3 Real Qwen Multi-Stage "
            "Cancellation Boundary"
        )
        print(
            "-------------------------------------------"
        )
        print(f"target expert:       {expert_id}")
        print(f"Stage 1 executions:  {stage1_calls}")
        print(f"Stage 2 executions:  {stage2_calls}")
        print(
            f"final status:       "
            f"{engine.task_status(expert_id)}"
        )
        print(
            f"RAM resident:       "
            f"{ram.contains(expert_id)}"
        )
        print(
            f"GPU resident:       "
            f"{gpu.contains(expert_id)}"
        )
        print(
            f"unfinished tasks:   "
            f"{engine.unfinished_tasks}"
        )

    finally:
        pipeline.stop()
        engine.stop()

        for expert_id in list(source_payloads):
            if gpu.contains(expert_id):
                gpu.remove(expert_id)

        if root.exists():
            shutil.rmtree(root)
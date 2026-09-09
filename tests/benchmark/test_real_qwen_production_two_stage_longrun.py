from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest
import torch
from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

from predictive_cache import CacheConfig, PredictiveExpertCache
from predictive_cache.prefetch.engine import PrefetchEngine
from predictive_cache.prefetch.pipeline import PrefetchPipeline
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


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="7-O-6 requires CUDA",
)


class MemoryExpertStore(ExpertStore):
    def __init__(self) -> None:
        self._records: dict[int, ExpertRecord] = {}

    def load(self, expert_id: int) -> None:
        return None

    def unload(self, expert_id: int) -> None:
        return None

    def prefetch(self, expert_id: int) -> None:
        return None

    def location(self, expert_id: int) -> ExpertLocation:
        if expert_id not in self._records:
            raise KeyError(expert_id)
        return ExpertLocation.RAM

    def put(self, record: ExpertRecord) -> None:
        record.location = ExpertLocation.RAM
        self._records[int(record.expert_id)] = record

    def get(self, expert_id: int) -> ExpertRecord | None:
        return self._records.get(int(expert_id))

    def contains(self, expert_id: int) -> bool:
        return int(expert_id) in self._records

    def remove(self, expert_id: int) -> None:
        self._records.pop(int(expert_id), None)


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
    ).eval()


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
            input_ids=input_ids,
            step=step,
        )

    events = capture.bridge.collector.events()[before:]

    current: list[int] = []

    for event in events:
        if event.step != step:
            continue

        current.extend(int(x) for x in event.expert_ids)

    return sorted(set(current))


def _wait_terminal(
    engine: PrefetchEngine,
    expert_ids: list[int],
    timeout_s: float = 10.0,
) -> None:
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


def test_real_qwen_two_stage_longrun_stability() -> None:
    torch.manual_seed(7)

    warmup_steps = 32
    longrun_steps = 1000

    root = Path(".pytest_nvme_qwen_two_stage_longrun")

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
    ram = MemoryExpertStore()
    gpu = GPUExpertStore("cuda")

    bridge = QwenRoutingBridge(cache)
    capture = QwenMoeRoutingCapture(bridge)

    payload = torch.randn(
        2048,
        1024,
        dtype=torch.float32,
    ).contiguous()

    scheduled_total = 0
    submitted_total = 0
    completed_total = 0
    failed_total = 0
    cancelled_total = 0
    prediction_steps = 0

    # Every expert written to NVMe gets one immutable reference tensor.
    source_payloads: dict[int, torch.Tensor] = {}

    def fallback_loader(task: PrefetchTask) -> None:
        raise AssertionError(
            "fallback loader was invoked; "
            "production transfer executor did not route "
            "NVMe -> RAM"
        )

    transfer = create_storage_transfer_executor(
        nvme,
        ram,
        fallback_loader,
        gpu=gpu,
    )

    engine = PrefetchEngine(
        loader=fallback_loader,
        max_queue_size=16,
        transfer_executor=transfer,
    )

    pipeline = PrefetchPipeline(
        scheduler,
        engine,
        auto_start=True,
    )

    try:
        # --------------------------------------------------------------
        # Warm-up: establish real Qwen prediction history.
        # --------------------------------------------------------------
        for step in range(warmup_steps):
            input_ids = torch.tensor(
                [[10 + (step % 118)]],
                dtype=torch.long,
            )

            current = _capture_current(
                capture,
                model,
                input_ids,
                step,
            )

            if current:
                scheduler.plan_prefetch_predictions(current)

        # --------------------------------------------------------------
        # Long-run real Qwen routing.
        # --------------------------------------------------------------
        for step in range(longrun_steps):
            input_ids = torch.tensor(
                [[10 + ((warmup_steps + step) % 118)]],
                dtype=torch.long,
            )

            current = _capture_current(
                capture,
                model,
                input_ids,
                warmup_steps + step,
            )

            if not current:
                continue

            candidates = scheduler.plan_prefetch_predictions(current)

            if not candidates:
                continue

            prediction_steps += 1

            predicted_ids = sorted(
                {
                    int(request.expert_id)
                    for request in candidates
                }
            )

            # Materialize every predicted expert on real NVMe.
            for expert_id in predicted_ids:
                if expert_id not in source_payloads:
                    expert_payload = payload.clone()

                    nvme.put(
                        ExpertRecord(
                            expert_id=expert_id,
                            location=ExpertLocation.NVME,
                            payload=expert_payload,
                        )
                    )

                    source_payloads[expert_id] = expert_payload.clone()

            # ----------------------------------------------------------
            # Real Pipeline submission.
            # ----------------------------------------------------------
            result = pipeline.process(current)

            scheduled_total += len(result.scheduled_requests)
            submitted_total += len(result.submitted_tasks)

            submitted_ids = [
                int(task.expert_id)
                for task in result.submitted_tasks
            ]

            if not submitted_ids:
                continue

            _wait_terminal(
                engine,
                submitted_ids,
            )

            completed_ids = [
                expert_id
                for expert_id in submitted_ids
                if engine.task_status(expert_id)
                == PrefetchStatus.COMPLETED
            ]

            failed_ids = [
                expert_id
                for expert_id in submitted_ids
                if engine.task_status(expert_id)
                == PrefetchStatus.FAILED
            ]

            cancelled_ids = [
                expert_id
                for expert_id in submitted_ids
                if engine.task_status(expert_id)
                == PrefetchStatus.CANCELLED
            ]

            # Every submitted task must be terminal.
            assert (
                len(completed_ids)
                + len(failed_ids)
                + len(cancelled_ids)
                == len(submitted_ids)
            )

            # This benchmark intentionally expects zero failures.
            assert failed_ids == []
            assert cancelled_ids == []

            completed_total += len(completed_ids)
            failed_total += len(failed_ids)
            cancelled_total += len(cancelled_ids)

            torch.cuda.synchronize()

            # ----------------------------------------------------------
            # Verify final GPU residency and payload integrity.
            # ----------------------------------------------------------
            for expert_id in completed_ids:
                assert ram.contains(expert_id)
                assert gpu.contains(expert_id)

                ram_record = ram.get(expert_id)
                gpu_record = gpu.get(expert_id)

                assert ram_record is not None
                assert gpu_record is not None

                assert isinstance(ram_record.payload, torch.Tensor)
                assert isinstance(gpu_record.payload, torch.Tensor)

                assert ram_record.payload.device.type == "cpu"
                assert gpu_record.payload.device.type == "cuda"

                assert torch.equal(
                    ram_record.payload,
                    source_payloads[expert_id],
                )

                assert torch.equal(
                    gpu_record.payload.cpu(),
                    source_payloads[expert_id],
                )

            # No task may remain unfinished between benchmark steps.
            assert engine.unfinished_tasks == 0

        torch.cuda.synchronize()

        # --------------------------------------------------------------
        # Global accounting invariants.
        # --------------------------------------------------------------
        assert submitted_total == completed_total
        assert failed_total == 0
        assert cancelled_total == 0

        assert engine.unfinished_tasks == 0

        print()
        print("7-P-5 Real Qwen -> Production NVMe -> RAM -> GPU Long-Run")
        print("-----------------------------------------------------------")
        print(f"warm-up steps:          {warmup_steps}")
        print(f"long-run routing steps: {longrun_steps}")
        print(f"steps with prediction:  {prediction_steps}")
        print(f"scheduled requests:     {scheduled_total}")
        print(f"submitted prefetches:   {submitted_total}")
        print(f"completed logical:      {completed_total}")
        print(f"failed:                 {failed_total}")
        print(f"cancelled:              {cancelled_total}")
        print(f"unfinished tasks:       {engine.unfinished_tasks}")
        print(f"NVMe expert files:      {len(source_payloads)}")

    finally:
        pipeline.stop()
        engine.stop()

        for expert_id in list(source_payloads):
            if gpu.contains(expert_id):
                gpu.remove(expert_id)

        if root.exists():
            shutil.rmtree(root)
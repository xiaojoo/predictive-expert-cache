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
    create_nvme_to_ram_handler,
    create_ram_to_gpu_handler,
)
from predictive_cache.prefetch.types import (
    PrefetchSource,
    PrefetchTarget,
    PrefetchTask,
    PrefetchStatus,
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
    reason="7-O-4 requires CUDA",
)


class MemoryExpertStore(ExpertStore):
    """Test-only RAM tier backed by process memory."""

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
    config = Qwen2MoeConfig(
        num_hidden_layers=2,
        hidden_size=128,
        intermediate_size=256,
        num_local_experts=8,
        num_experts_per_tok=2,
        vocab_size=128,
    )
    return Qwen2MoeForCausalLM(config).eval()


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
        states = [
            engine.task_status(expert_id)
            for expert_id in expert_ids
        ]

        if all(
            state in {
                PrefetchStatus.COMPLETED,
                PrefetchStatus.FAILED,
                PrefetchStatus.CANCELLED,
            }
            for state in states
        ):
            return

        time.sleep(0.001)

    raise AssertionError(
        f"prefetch tasks did not reach terminal state: {expert_ids}"
    )


def test_real_qwen_prefetch_nvme_ram_gpu_pipeline() -> None:
    torch.manual_seed(7)

    root = Path(".pytest_nvme_qwen_pipeline")
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

    original_payloads: dict[int, torch.Tensor] = {}

    # The real transfer handlers are used by the pipeline executor.
    nvme_to_ram = create_nvme_to_ram_handler(nvme, ram)
    ram_to_gpu = create_ram_to_gpu_handler(ram, gpu)

    def loader(task: PrefetchTask) -> None:
        if task.source == PrefetchSource.NVME:
            nvme_to_ram(task)
            return

        if task.source == PrefetchSource.RAM:
            ram_to_gpu(task)
            return

        raise AssertionError(
            f"unexpected prefetch source: {task.source}"
        )

    engine = PrefetchEngine(
        loader=loader,
        max_queue_size=16,
    )

    pipeline = PrefetchPipeline(
        scheduler,
        engine,
        auto_start=True,
    )

    try:
        # ------------------------------------------------------------------
        # 1. Warm up the real Qwen routing predictor until predictions exist.
        # ------------------------------------------------------------------
        prediction_current: list[int] | None = None
        predicted: list[int] = []

        for step in range(32):
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

            if not current:
                continue

            candidates = scheduler.plan_prefetch_predictions(current)

            if candidates:
                prediction_current = current
                predicted = sorted(
                    {
                        int(request.expert_id)
                        for request in candidates
                    }
                )
                break

        assert prediction_current is not None
        assert predicted

        # ------------------------------------------------------------------
        # 2. Put actual expert payloads into the real NVMe tier.
        #
        #    The payload is intentionally 8 MiB so the benchmark exercises
        #    an actual filesystem-backed transfer rather than a tiny object.
        # ------------------------------------------------------------------
        payload = torch.randn(
            2048,
            1024,
            dtype=torch.float32,
        ).contiguous()

        for expert_id in predicted:
            expert_payload = payload.clone()

            nvme.put(
                ExpertRecord(
                    expert_id=expert_id,
                    location=ExpertLocation.NVME,
                    payload=expert_payload,
                )
            )

            original_payloads[expert_id] = expert_payload.clone()

        # ------------------------------------------------------------------
        # 3. Real prediction -> scheduler -> pipeline.
        # ------------------------------------------------------------------
        result = pipeline.process(prediction_current)

        assert result.scheduled_requests
        assert result.submitted_tasks

        submitted_ids = [
            int(task.expert_id)
            for task in result.submitted_tasks
        ]

        assert submitted_ids

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

        assert failed_ids == []
        assert cancelled_ids == []
        assert completed_ids

        # ------------------------------------------------------------------
        # 4. Explicitly complete the second physical hop for each completed
        #    expert: RAM -> GPU.
        # ------------------------------------------------------------------
        gpu_ids: list[int] = []

        for expert_id in completed_ids:
            task = PrefetchTask(
                expert_id=expert_id,
                source=PrefetchSource.RAM,
                target=PrefetchTarget.GPU,
            )

            ram_to_gpu(task)
            gpu_ids.append(expert_id)

        torch.cuda.synchronize()

        assert gpu_ids == completed_ids

        # ------------------------------------------------------------------
        # 5. Verify actual residency and payload integrity.
        # ------------------------------------------------------------------
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
                gpu_record.payload.cpu(),
                original_payloads[expert_id],
            )

        # ------------------------------------------------------------------
        # 6. Real later demand:
        #
        #    Ask the cache for an actually prefetched expert. Because the
        #    GPU tier is already resident, this must not require another
        #    NVMe/RAM transfer.
        # ------------------------------------------------------------------
        demand_expert = completed_ids[0]

        before_gpu = gpu.contains(demand_expert)
        assert before_gpu

        demand_record = gpu.get(demand_expert)

        assert demand_record is not None
        assert isinstance(demand_record.payload, torch.Tensor)
        assert demand_record.payload.device.type == "cuda"

        torch.cuda.synchronize()

        # ------------------------------------------------------------------
        # 7. Accounting invariants.
        # ------------------------------------------------------------------
        terminal = (
            len(completed_ids)
            + len(failed_ids)
            + len(cancelled_ids)
        )

        assert terminal == len(result.submitted_tasks)
        assert engine.unfinished_tasks == 0

        print()
        print("7-O-4 Real Qwen -> NVMe -> RAM -> GPU Pipeline")
        print("------------------------------------------------")
        print(f"prediction current:       {prediction_current}")
        print(f"predicted candidates:     {predicted}")
        print(f"scheduled requests:       {len(result.scheduled_requests)}")
        print(f"submitted prefetches:     {len(result.submitted_tasks)}")
        print(f"completed NVMe -> RAM:    {len(completed_ids)}")
        print(f"failed:                   {len(failed_ids)}")
        print(f"cancelled:                {len(cancelled_ids)}")
        print(f"GPU resident:             {len(gpu_ids)}")
        print(f"demand expert:            {demand_expert}")
        print(f"demand GPU hit:           {before_gpu}")
        print(f"unfinished tasks:         {engine.unfinished_tasks}")

    finally:

        pipeline.stop()

        engine.stop()

        for expert_id in gpu_ids:
            gpu.remove(expert_id)

        if root.exists():
            shutil.rmtree(root)
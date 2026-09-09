from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest
import torch
from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

from predictive_cache import CacheConfig, PredictiveExpertCache
from predictive_cache.prefetch.admission import AdmissionReason
from predictive_cache.prefetch.engine import PrefetchEngine
from predictive_cache.prefetch.pipeline import PrefetchPipeline
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
    reason="7-Q-4 requires CUDA",
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


def test_real_qwen_pipeline_failure_then_retry() -> None:
    torch.manual_seed(7)

    warmup_steps = 32

    root = Path(
        ".pytest_nvme_qwen_pipeline_failure_recovery"
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
    ram = MemoryExpertStore()
    gpu = GPUExpertStore("cuda")

    bridge = QwenRoutingBridge(cache)
    capture = QwenMoeRoutingCapture(bridge)

    source_payloads: dict[int, torch.Tensor] = {}

    try:
        # ----------------------------------------------------------
        # Establish real Qwen prediction history.
        # ----------------------------------------------------------
        selected_current: list[int] = []
        selected_candidates: list[int] = []

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

            if not current:
                continue

            candidates = (
                scheduler.plan_prefetch_predictions(
                    current
                )
            )

            if candidates:
                selected_current = list(current)

                selected_candidates = sorted(
                    {
                        int(request.expert_id)
                        for request in candidates
                    }
                )

        assert selected_current
        assert selected_candidates

        # ----------------------------------------------------------
        # Materialize every predicted expert on real NVMe.
        # ----------------------------------------------------------
        payload = torch.randn(
            2048,
            1024,
            dtype=torch.float32,
        ).contiguous()

        for expert_id in selected_candidates:
            expert_payload = payload.clone()

            nvme.put(
                ExpertRecord(
                    expert_id=expert_id,
                    location=ExpertLocation.NVME,
                    payload=expert_payload,
                )
            )

            source_payloads[expert_id] = (
                expert_payload.clone()
            )

        # ----------------------------------------------------------
        # Production two-stage transfer chain.
        #
        # Stage 1:
        #     NVMe -> RAM
        #
        # Stage 2:
        #     RAM -> GPU
        #
        # Stage 1 deliberately fails on the first attempt for
        # every submitted expert. The real Stage-1 handler is
        # used on retry.
        # ----------------------------------------------------------
        real_nvme_to_ram = create_nvme_to_ram_handler(
            nvme,
            ram,
        )

        real_ram_to_gpu = create_ram_to_gpu_handler(
            ram,
            gpu,
        )

        attempts = {
            "stage_one": 0,
        }

        first_attempt_count = len(
            selected_candidates
        )

        def flaky_nvme_to_ram(
            task: PrefetchTask,
        ) -> None:
            attempts["stage_one"] += 1

            if (
                attempts["stage_one"]
                <= first_attempt_count
            ):
                raise RuntimeError(
                    "injected production-pipeline "
                    "NVMe failure"
                )

            real_nvme_to_ram(task)

        chain = PrefetchStageChain(
            [
                PrefetchStage(
                    source=PrefetchSource.NVME,
                    target=PrefetchTarget.RAM,
                    handler=flaky_nvme_to_ram,
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
                    "fallback loader was invoked"
                )
            )
        )

        transfer.register(
            PrefetchSource.NVME,
            PrefetchTarget.RAM,
            chain,
        )

        engine = PrefetchEngine(
            loader=lambda task: (_ for _ in ()).throw(
                AssertionError(
                    "fallback loader was invoked"
                )
            ),
            max_queue_size=16,
            transfer_executor=transfer,
        )

        pipeline = PrefetchPipeline(
            scheduler,
            engine,
            auto_start=True,
        )

        try:
            # ------------------------------------------------------
            # Attempt 1
            #
            # Real Qwen routing
            # -> scheduler
            # -> admission
            # -> engine
            # -> production two-stage transfer
            # -> injected NVMe -> RAM failure
            # -> FAILED
            # ------------------------------------------------------
            first = pipeline.process(
                selected_current
            )

            first_ids = [
                int(task.expert_id)
                for task in first.submitted_tasks
            ]

            assert first_ids

            assert set(first_ids).issubset(
                set(selected_candidates)
            )

            _wait_terminal(
                engine,
                first_ids,
            )

            # Every first attempt must fail.
            assert all(
                engine.task_status(expert_id)
                == PrefetchStatus.FAILED
                for expert_id in first_ids
            )

            # No first-attempt task may complete or cancel.
            assert len(
                engine.completed_tasks
            ) == 0

            assert len(
                engine.failed_tasks
            ) == len(first_ids)

            assert len(
                engine.cancelled_tasks
            ) == 0

            # FAILED tasks must leave the unfinished set.
            assert engine.unfinished_tasks == 0

            # Stage 1 failed before NVMe -> RAM.
            assert all(
                not ram.contains(expert_id)
                for expert_id in first_ids
            )

            # Therefore Stage 2 must never have produced GPU data.
            assert all(
                gpu.get(expert_id) is None
                for expert_id in first_ids
            )

            # ------------------------------------------------------
            # Critical lifecycle assertion:
            #
            # A FAILED task must not remain in either the queued
            # or running duplicate-suppression sets.
            # ------------------------------------------------------
            assert all(
                expert_id not in engine.running_tasks
                for expert_id in first_ids
            )

            assert all(
                expert_id not in engine.queued_tasks
                for expert_id in first_ids
            )

            # ------------------------------------------------------
            # Attempt 2
            #
            # Submit the SAME real Qwen current state through the
            # COMPLETE production pipeline again.
            #
            # The retry must be admitted instead of being rejected
            # as a queued/running duplicate.
            # ------------------------------------------------------
            second = pipeline.process(
                selected_current
            )

            second_ids = [
                int(task.expert_id)
                for task in second.submitted_tasks
            ]

            assert second_ids

            assert set(second_ids) == set(first_ids)

            assert second.admission_decisions

            # Retry must not be rejected because the previous task
            # is still considered queued.
            assert all(
                decision.reason
                != AdmissionReason.REJECT_DUPLICATE_QUEUED
                for decision in second.admission_decisions
            )

            # Retry must not be rejected because the previous task
            # is still considered running.
            assert all(
                decision.reason
                != AdmissionReason.REJECT_DUPLICATE_RUNNING
                for decision in second.admission_decisions
            )

            # The submitted retry set itself proves that these
            # requests passed Admission and Engine.submit().
            assert len(second_ids) == len(
                second.admission_decisions
            )

            _wait_terminal(
                engine,
                second_ids,
            )

            # Every retry must complete successfully.
            assert all(
                engine.task_status(expert_id)
                == PrefetchStatus.COMPLETED
                for expert_id in second_ids
            )

            assert len(
                engine.completed_tasks
            ) == len(second_ids)

            assert len(
                engine.failed_tasks
            ) == 0

            assert len(
                engine.cancelled_tasks
            ) == 0

            assert engine.unfinished_tasks == 0

            # Exactly one failed Stage-1 attempt followed by one
            # successful Stage-1 attempt for every expert.
            assert attempts["stage_one"] == (
                len(first_ids)
                + len(second_ids)
            )

            torch.cuda.synchronize()

            # ------------------------------------------------------
            # Final physical residency + payload integrity.
            #
            # Stage 1:
            #     NVMe -> RAM
            #
            # Stage 2:
            #     RAM -> GPU
            # ------------------------------------------------------
            for expert_id in second_ids:
                assert ram.contains(expert_id)
                assert gpu.contains(expert_id)

                ram_record = ram.get(expert_id)
                gpu_record = gpu.get(expert_id)

                assert ram_record is not None
                assert gpu_record is not None

                assert isinstance(
                    ram_record.payload,
                    torch.Tensor,
                )

                assert isinstance(
                    gpu_record.payload,
                    torch.Tensor,
                )

                assert (
                    ram_record.payload.device.type
                    == "cpu"
                )

                assert (
                    gpu_record.payload.device.type
                    == "cuda"
                )

                assert torch.equal(
                    ram_record.payload,
                    source_payloads[expert_id],
                )

                assert torch.equal(
                    gpu_record.payload.cpu(),
                    source_payloads[expert_id],
                )

            print()
            print(
                "7-Q-4 Real Qwen Pipeline Failure Recovery"
            )
            print(
                "-----------------------------------------"
            )
            print(
                f"warm-up steps:        {warmup_steps}"
            )
            print(
                f"prediction current:   {selected_current}"
            )
            print(
                f"predicted candidates: {selected_candidates}"
            )
            print(
                f"first submitted:      {len(first_ids)}"
            )
            print(
                f"first failed:         {len(first_ids)}"
            )
            print(
                f"retry submitted:      {len(second_ids)}"
            )
            print(
                f"retry completed:      {len(second_ids)}"
            )
            print(
                f"stage-1 attempts:     "
                f"{attempts['stage_one']}"
            )
            print(
                f"unfinished tasks:     "
                f"{engine.unfinished_tasks}"
            )

        finally:
            pipeline.stop()

    finally:
        if root.exists():
            shutil.rmtree(root)
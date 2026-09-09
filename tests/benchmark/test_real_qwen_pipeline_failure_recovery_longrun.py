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
    reason="7-Q-5 requires CUDA",
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


def test_real_qwen_pipeline_failure_recovery_longrun() -> None:
    torch.manual_seed(7)

    warmup_steps = 32
    longrun_steps = 100

    root = Path(
        ".pytest_nvme_qwen_pipeline_failure_recovery_longrun"
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

    # --------------------------------------------------------------
    # Global accounting.
    # --------------------------------------------------------------
    routing_steps = 0
    prediction_steps = 0

    scheduled_total = 0
    submitted_total = 0

    first_attempt_total = 0
    first_failed_total = 0

    retry_submitted_total = 0
    retry_completed_total = 0

    injected_failure_steps = 0

    try:
        # ----------------------------------------------------------
        # Warm-up.
        # ----------------------------------------------------------
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
                scheduler.plan_prefetch_predictions(
                    current
                )

        # ----------------------------------------------------------
        # Real production transfer handlers.
        # ----------------------------------------------------------
        real_nvme_to_ram = create_nvme_to_ram_handler(
            nvme,
            ram,
        )

        real_ram_to_gpu = create_ram_to_gpu_handler(
            ram,
            gpu,
        )

        # Each benchmark step gets its own failure budget.
        failure_budget = {
            "remaining": 0,
        }

        def flaky_nvme_to_ram(
            task: PrefetchTask,
        ) -> None:
            if failure_budget["remaining"] > 0:
                failure_budget["remaining"] -= 1

                raise RuntimeError(
                    "injected long-run NVMe -> RAM failure"
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

        def fallback_loader(
            task: PrefetchTask,
        ) -> None:
            raise AssertionError(
                "fallback loader was invoked"
            )

        transfer = PrefetchTransferExecutor(
            fallback_loader
        )

        transfer.register(
            PrefetchSource.NVME,
            PrefetchTarget.RAM,
            chain,
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
            # ------------------------------------------------------
            # Long-run real Qwen routing.
            #
            # Every 10th prediction step injects a failure for
            # every submitted task in that step.
            # ------------------------------------------------------
            for step in range(longrun_steps):
                absolute_step = warmup_steps + step

                input_ids = torch.tensor(
                    [[10 + (absolute_step % 118)]],
                    dtype=torch.long,
                )

                current = _capture_current(
                    capture,
                    model,
                    input_ids,
                    absolute_step,
                )

                routing_steps += 1

                if not current:
                    continue

                candidates = (
                    scheduler.plan_prefetch_predictions(
                        current
                    )
                )

                if not candidates:
                    continue

                prediction_steps += 1

                predicted_ids = sorted(
                    {
                        int(request.expert_id)
                        for request in candidates
                    }
                )

                # --------------------------------------------------
                # Materialize predicted experts on real NVMe.
                # --------------------------------------------------
                for expert_id in predicted_ids:
                    if expert_id in source_payloads:
                        continue

                    expert_payload = torch.randn(
                        2048,
                        1024,
                        dtype=torch.float32,
                    ).contiguous()

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

                # --------------------------------------------------
                # Every 10th prediction step:
                #
                # force all first attempts to fail.
                # --------------------------------------------------
                inject_failure = (
                    prediction_steps % 10 == 0
                )

                if inject_failure:
                    injected_failure_steps += 1

                if inject_failure:
                    # ----------------------------------------------------------
                    # Remove stale residency from earlier long-run iterations.
                    #
                    # The same expert IDs can be predicted repeatedly.  We need
                    # the failure step to start from a genuinely cold residency
                    # state so that "Stage 1 failed => no RAM/GPU residency"
                    # is a meaningful assertion.
                    # ----------------------------------------------------------
                    for expert_id in predicted_ids:
                        if ram.contains(expert_id):
                            ram.remove(expert_id)

                        if gpu.contains(expert_id):
                            gpu.remove(expert_id)

                    failure_budget["remaining"] = len(
                        predicted_ids
                    )

                first = pipeline.process(current)

                scheduled_total += len(
                    first.scheduled_requests
                )

                submitted_total += len(
                    first.submitted_tasks
                )

                first_ids = [
                    int(task.expert_id)
                    for task in first.submitted_tasks
                ]

                if not first_ids:
                    failure_budget["remaining"] = 0
                    continue

                first_attempt_total += len(first_ids)

                _wait_terminal(
                    engine,
                    first_ids,
                )

                # --------------------------------------------------
                # Normal step:
                # all tasks must complete immediately.
                # --------------------------------------------------
                if not inject_failure:
                    assert all(
                        engine.task_status(expert_id)
                        == PrefetchStatus.COMPLETED
                        for expert_id in first_ids
                    )

                    assert engine.unfinished_tasks == 0

                    for expert_id in first_ids:
                        assert ram.contains(expert_id)
                        assert gpu.contains(expert_id)

                    continue

                # --------------------------------------------------
                # Failure step:
                # every first attempt must fail.
                # --------------------------------------------------
                assert all(
                    engine.task_status(expert_id)
                    == PrefetchStatus.FAILED
                    for expert_id in first_ids
                )

                first_failed_total += len(first_ids)

                assert set(first_ids).issubset(
                    set(engine.failed_tasks)
                )

                assert not any(
                    expert_id in engine.completed_tasks
                    for expert_id in first_ids
                )

                assert not any(
                    expert_id in engine.cancelled_tasks
                    for expert_id in first_ids
                )

                assert engine.unfinished_tasks == 0

                # Stage 1 failed, so Stage 2 must not run and no
                # new residency should exist for these cold-start tasks.
                assert all(
                    not ram.contains(expert_id)
                    for expert_id in first_ids
                )

                assert all(
                    not gpu.contains(expert_id)
                    for expert_id in first_ids
                )

                # --------------------------------------------------
                # Critical recovery invariant.
                # --------------------------------------------------
                assert all(
                    expert_id not in engine.queued_tasks
                    for expert_id in first_ids
                )

                assert all(
                    expert_id not in engine.running_tasks
                    for expert_id in first_ids
                )

                # --------------------------------------------------
                # Retry the SAME real Qwen current state through
                # the complete production pipeline.
                #
                # Failure budget is now zero, so real Stage 1
                # executes, followed by real Stage 2.
                # --------------------------------------------------
                failure_budget["remaining"] = 0

                retry = pipeline.process(current)

                retry_submitted_ids = [
                    int(task.expert_id)
                    for task in retry.submitted_tasks
                ]

                assert retry_submitted_ids

                assert set(
                    retry_submitted_ids
                ) == set(first_ids)

                assert retry.admission_decisions

                # No retry may be rejected as a stale queued
                # duplicate.
                assert all(
                    decision.reason
                    != AdmissionReason.REJECT_DUPLICATE_QUEUED
                    for decision in retry.admission_decisions
                )

                # No retry may be rejected as a stale running
                # duplicate.
                assert all(
                    decision.reason
                    != AdmissionReason.REJECT_DUPLICATE_RUNNING
                    for decision in retry.admission_decisions
                )

                assert len(
                    retry_submitted_ids
                ) == len(
                    retry.admission_decisions
                )

                retry_submitted_total += len(
                    retry_submitted_ids
                )

                _wait_terminal(
                    engine,
                    retry_submitted_ids,
                )

                # Every retry must succeed.
                assert all(
                    engine.task_status(expert_id)
                    == PrefetchStatus.COMPLETED
                    for expert_id in retry_submitted_ids
                )

                retry_completed_total += len(
                    retry_submitted_ids
                )

                assert set(retry_submitted_ids).issubset(
                    set(engine.completed_tasks)
                )

                assert not any(
                    expert_id in engine.failed_tasks
                    for expert_id in retry_submitted_ids
                )

                assert not any(
                    expert_id in engine.cancelled_tasks
                    for expert_id in retry_submitted_ids
                )

                assert engine.unfinished_tasks == 0

                torch.cuda.synchronize()

                # --------------------------------------------------
                # Physical residency and payload integrity.
                # --------------------------------------------------
                for expert_id in retry_submitted_ids:
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

                # --------------------------------------------------
                # No state contamination between benchmark steps.
                # --------------------------------------------------
                assert engine.unfinished_tasks == 0

            torch.cuda.synchronize()

            # ------------------------------------------------------
            # Global invariants.
            # ------------------------------------------------------
            assert routing_steps == longrun_steps

            assert prediction_steps > 0

            assert injected_failure_steps > 0

            assert first_attempt_total > 0

            assert first_failed_total == (
                retry_completed_total
            )

            assert retry_submitted_total == (
                retry_completed_total
            )

            # Every submitted first attempt eventually either
            # completed normally or failed and was retried.
            assert submitted_total == (
                first_attempt_total
            )

            assert engine.unfinished_tasks == 0

            assert len(
                engine.queued_tasks
            ) == 0

            assert len(
                engine.running_tasks
            ) == 0

            print()
            print(
                "7-Q-5 Real Qwen Pipeline Failure Recovery "
                "Long-Run"
            )
            print(
                "------------------------------------------------"
            )
            print(
                f"warm-up steps:          {warmup_steps}"
            )
            print(
                f"long-run routing:      {routing_steps}"
            )
            print(
                f"prediction steps:      {prediction_steps}"
            )
            print(
                f"failure steps:         "
                f"{injected_failure_steps}"
            )
            print(
                f"scheduled requests:    {scheduled_total}"
            )
            print(
                f"first submissions:     {first_attempt_total}"
            )
            print(
                f"first failures:        {first_failed_total}"
            )
            print(
                f"retry submissions:     {retry_submitted_total}"
            )
            print(
                f"retry completions:     {retry_completed_total}"
            )
            print(
                f"unfinished tasks:      "
                f"{engine.unfinished_tasks}"
            )
            print(
                f"queued tasks:          "
                f"{len(engine.queued_tasks)}"
            )
            print(
                f"running tasks:         "
                f"{len(engine.running_tasks)}"
            )
            print(
                f"NVMe experts:          "
                f"{len(source_payloads)}"
            )

        finally:
            pipeline.stop()

    finally:
        if root.exists():
            shutil.rmtree(root)

from __future__ import annotations

import shutil
import threading
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
from predictive_cache.storage.expert_store import ExpertLocation
from predictive_cache.storage.gpu import GPUExpertStore
from predictive_cache.storage.nvme import NVMeExpertStore
from predictive_cache.storage.ram import RAMExpertStore


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="7-R-2 requires CUDA",
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


def _wait_for(
    predicate,
    *,
    timeout_s: float = 10.0,
    description: str,
) -> None:
    deadline = time.perf_counter() + timeout_s

    while time.perf_counter() < deadline:
        if predicate():
            return

        time.sleep(0.001)

    raise AssertionError(
        f"timeout waiting for {description}"
    )


def test_real_qwen_pipeline_backpressure() -> None:
    torch.manual_seed(17)

    warmup_steps = 32
    pressure_steps = 100

    root = Path(
        ".pytest_nvme_qwen_pipeline_backpressure"
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

    source_payloads: dict[int, torch.Tensor] = {}

    routing_steps = 0
    prediction_steps = 0

    scheduled_total = 0
    accepted_total = 0
    rejected_total = 0

    queue_full_rejections = 0
    duplicate_rejections = 0

    executed_ids: list[int] = []
    execution_counts: dict[int, int] = {}

    first_started = threading.Event()
    release_first = threading.Event()

    try:
        # ----------------------------------------------------------
        # Warm-up real Qwen routing.
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
        # Real two-stage production transfer.
        #
        # The first Stage-1 task is deliberately blocked so that
        # the finite queue remains saturated while pipeline.process()
        # continues producing predictions.
        # ----------------------------------------------------------
        real_nvme_to_ram = create_nvme_to_ram_handler(
            nvme,
            ram,
        )

        real_ram_to_gpu = create_ram_to_gpu_handler(
            ram,
            gpu,
        )

        blocked_first_id: int | None = None

        def backpressure_nvme_to_ram(
            task: PrefetchTask,
        ) -> None:
            nonlocal blocked_first_id

            if blocked_first_id is None:
                blocked_first_id = int(task.expert_id)
                first_started.set()

                release_first.wait(timeout=15.0)

            real_nvme_to_ram(task)

            executed_ids.append(
                int(task.expert_id)
            )

            execution_counts[int(task.expert_id)] = (
                execution_counts.get(int(task.expert_id), 0)
                + 1
            )

        chain = PrefetchStageChain(
            [
                PrefetchStage(
                    source=PrefetchSource.NVME,
                    target=PrefetchTarget.RAM,
                    handler=backpressure_nvme_to_ram,
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

        # Queue capacity intentionally tiny.
        engine = PrefetchEngine(
            loader=fallback_loader,
            num_workers=1,
            max_queue_size=2,
            transfer_executor=transfer,
        )

        pipeline = PrefetchPipeline(
            scheduler,
            engine,
            auto_start=True,
        )

        try:
            # ------------------------------------------------------
            # Produce real prediction traffic until the first task
            # is RUNNING and the queue becomes full.
            # ------------------------------------------------------
            first_current: list[int] = []

            for step in range(pressure_steps):
                absolute_step = warmup_steps + step

                input_ids = torch.tensor(
                    [[10 + (absolute_step % 118)]],
                    dtype=torch.long,
                    device="cuda",
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

                for expert_id in predicted_ids:
                    if expert_id in source_payloads:
                        continue

                    payload = torch.randn(
                        256,
                        256,
                        dtype=torch.float32,
                    ).contiguous()

                    nvme.put(
                        ExpertRecord(
                            expert_id=expert_id,
                            location=ExpertLocation.NVME,
                            payload=payload,
                        )
                    )

                    source_payloads[expert_id] = (
                        payload.clone()
                    )

                result = pipeline.process(current)

                scheduled_total += len(
                    result.scheduled_requests
                )

                accepted_total += len(
                    result.submitted_tasks
                )

                for decision in result.admission_decisions:
                    if decision.reason == (
                        AdmissionReason.REJECT_QUEUE_FULL
                    ):
                        queue_full_rejections += 1

                    if decision.reason in {
                        AdmissionReason.REJECT_DUPLICATE_QUEUED,
                        AdmissionReason.REJECT_DUPLICATE_RUNNING,
                    }:
                        duplicate_rejections += 1

                rejected_total += sum(
                    not decision.admitted
                    for decision in result.admission_decisions
                )

                if first_started.is_set():
                    first_current = current
                    break

            assert prediction_steps > 0

            assert first_started.wait(timeout=5.0)

            # ------------------------------------------------------
            # We need a genuinely saturated queue.
            # ------------------------------------------------------
            _wait_for(
                lambda: engine.queue_full,
                description="queue saturation",
            )

            assert engine.queue_size == 2

            queued_before_release = set(
                engine.queued_tasks
            )

            assert len(queued_before_release) == 2

            # ------------------------------------------------------
            # Explicit duplicate pressure against both RUNNING and
            # QUEUED tasks.
            #
            # These must be rejected by admission rather than
            # generating duplicate execution.
            # ------------------------------------------------------
            duplicate_result = pipeline.process(
                first_current
            )

            duplicate_reasons = {
                decision.reason
                for decision in duplicate_result.admission_decisions
            }

            assert (
                AdmissionReason.REJECT_DUPLICATE_RUNNING
                in duplicate_reasons
                or
                AdmissionReason.REJECT_DUPLICATE_QUEUED
                in duplicate_reasons
            )

            assert duplicate_result.submitted_tasks == []

            # ------------------------------------------------------
            # While saturated, at least one normal prediction must
            # encounter queue-full backpressure.
            # ------------------------------------------------------
            saturation_found = False

            for step in range(
                warmup_steps + routing_steps,
                warmup_steps + routing_steps + 20,
            ):
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

                if not current:
                    continue

                result = pipeline.process(current)

                scheduled_total += len(
                    result.scheduled_requests
                )

                accepted_total += len(
                    result.submitted_tasks
                )

                for decision in result.admission_decisions:
                    if decision.reason == (
                        AdmissionReason.REJECT_QUEUE_FULL
                    ):
                        queue_full_rejections += 1
                        saturation_found = True

                    if decision.reason in {
                        AdmissionReason.REJECT_DUPLICATE_QUEUED,
                        AdmissionReason.REJECT_DUPLICATE_RUNNING,
                    }:
                        duplicate_rejections += 1

                rejected_total += sum(
                    not decision.admitted
                    for decision in result.admission_decisions
                )

                if saturation_found:
                    break

            assert saturation_found
            assert queue_full_rejections > 0

            # ------------------------------------------------------
            # Cancellation must release one queue slot immediately.
            # ------------------------------------------------------
            cancel_id = next(
                iter(queued_before_release)
            )

            assert engine.cancel(cancel_id) is True

            assert (
                engine.task_status(cancel_id)
                == PrefetchStatus.CANCELLED
            )

            assert engine.queue_size == 1
            assert engine.queue_full is False

            # A released queue slot must be usable by the pipeline.
            #
            # The prediction may contain multiple candidates. Once the
            # first new candidate is admitted, the queue can immediately
            # become full again, so later candidates are legitimately
            # rejected by backpressure.
            fresh_result = pipeline.process(
                first_current
            )

            assert fresh_result.submitted_tasks

            assert any(
                decision.admitted
                for decision in fresh_result.admission_decisions
            )

            # At least one task must have crossed the admission gate
            # after cancellation released capacity.
            assert len(
                fresh_result.submitted_tasks
            ) >= 1

            # ------------------------------------------------------
            # Release the blocked first task and drain everything.
            # ------------------------------------------------------
            release_first.set()

            pipeline.stop()

            # ------------------------------------------------------
            # Final reclamation invariants.
            # ------------------------------------------------------
            assert engine.unfinished_tasks == 0
            assert engine.queue_size == 0
            assert len(engine.queued_tasks) == 0
            assert len(engine.running_tasks) == 0

            assert engine.running is False

            # Every physically executed task must have executed
            # exactly once. No duplicate execution is permitted.
            assert all(
                count == 1
                for count in execution_counts.values()
            )

            # Cancelled task must never execute.
            assert cancel_id not in executed_ids

            # Every successful task that reached the transfer chain
            # must have completed and reached both storage tiers.
            for expert_id in executed_ids:
                assert (
                    engine.task_status(expert_id)
                    == PrefetchStatus.COMPLETED
                )

                assert ram.contains(expert_id)
                assert gpu.contains(expert_id)

                ram_record = ram.get(expert_id)
                gpu_record = gpu.get(expert_id)

                assert ram_record is not None
                assert gpu_record is not None

                assert torch.equal(
                    ram_record.payload,
                    source_payloads[expert_id],
                )

                assert torch.equal(
                    gpu_record.payload.cpu(),
                    source_payloads[expert_id],
                )

            assert accepted_total >= len(executed_ids)

            print()
            print(
                "7-R-2 Real Qwen Production Backpressure"
            )
            print(
                "-------------------------------------------"
            )
            print(
                f"routing steps:          {routing_steps}"
            )
            print(
                f"prediction steps:      {prediction_steps}"
            )
            print(
                f"scheduled requests:    {scheduled_total}"
            )
            print(
                f"accepted submissions:  {accepted_total}"
            )
            print(
                f"rejected decisions:    {rejected_total}"
            )
            print(
                f"queue-full rejects:    {queue_full_rejections}"
            )
            print(
                f"duplicate rejects:     {duplicate_rejections}"
            )
            print(
                f"executed tasks:        {len(executed_ids)}"
            )
            print(
                f"cancelled task:        {cancel_id}"
            )
            print(
                f"unfinished tasks:      {engine.unfinished_tasks}"
            )
            print(
                f"queued tasks:          {len(engine.queued_tasks)}"
            )
            print(
                f"running tasks:         {len(engine.running_tasks)}"
            )

        finally:
            release_first.set()

            if pipeline.running:
                pipeline.stop()

    finally:
        if root.exists():
            shutil.rmtree(root)



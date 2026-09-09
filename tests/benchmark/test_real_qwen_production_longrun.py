from __future__ import annotations

import shutil
import time
from pathlib import Path
from threading import Event

import pytest
import torch
from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

from predictive_cache import CacheConfig, PredictiveExpertCache
from predictive_cache.prefetch.admission import AdmissionReason
from predictive_cache.prefetch.engine import PrefetchEngine
from predictive_cache.prefetch.pipeline import PrefetchPipeline
from predictive_cache.prefetch.stages import PrefetchStage, PrefetchStageChain
from predictive_cache.prefetch.storage import (
    create_nvme_to_ram_handler,
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
from predictive_cache.storage.expert_store import ExpertLocation
from predictive_cache.storage.gpu import GPUExpertStore
from predictive_cache.storage.nvme import NVMeExpertStore
from predictive_cache.storage.ram import RAMExpertStore


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="7-R-5 requires CUDA",
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

        current.extend(int(x) for x in event.expert_ids)

    return sorted(set(current))


def _wait_terminal(
    engine: PrefetchEngine,
    expert_id: int,
    timeout_s: float = 10.0,
) -> PrefetchStatus:
    deadline = time.perf_counter() + timeout_s

    while time.perf_counter() < deadline:
        status = engine.task_status(expert_id)

        if status in {
            PrefetchStatus.COMPLETED,
            PrefetchStatus.FAILED,
            PrefetchStatus.CANCELLED,
        }:
            return status

        time.sleep(0.001)

    raise AssertionError(
        f"task did not reach terminal state: {expert_id}"
    )


def _assert_idle(engine: PrefetchEngine) -> None:
    assert engine.unfinished_tasks == 0
    assert engine.queue_size == 0
    assert len(engine.queued_tasks) == 0
    assert len(engine.running_tasks) == 0


def _assert_payload_integrity(
    expert_id: int,
    ram: RAMExpertStore,
    gpu: GPUExpertStore,
    source_payloads: dict[int, torch.Tensor],
) -> None:
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


def test_real_qwen_production_longrun_1000_steps() -> None:
    torch.manual_seed(7)

    warmup_steps = 32
    longrun_steps = 1000

    root = Path(".pytest_nvme_qwen_production_longrun")

    if root.exists():
        shutil.rmtree(root)

    root.mkdir(parents=True)

    model = _make_model()

    cache = PredictiveExpertCache(
        CacheConfig(
            capacity=8,
            prediction_top_k=1,
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

    normal_completed = 0

    cancellation_steps = 0
    cancelled_total = 0

    failure_steps = 0
    failed_total = 0

    retry_submitted_total = 0
    retry_completed_total = 0

    try:
        # ----------------------------------------------------------
        # Warm-up.
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
                scheduler.plan_prefetch_predictions(current)

        # ----------------------------------------------------------
        # 1000-step real Qwen production run.
        # ----------------------------------------------------------
        for step in range(longrun_steps):
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

            candidates = scheduler.plan_prefetch_predictions(
                current
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

            # ------------------------------------------------------
            # Materialize predicted experts on real NVMe.
            # ------------------------------------------------------
            for expert_id in predicted_ids:
                if expert_id in source_payloads:
                    continue

                payload = torch.randn(
                    2048,
                    1024,
                    dtype=torch.float32,
                ).contiguous()

                nvme.put(
                    ExpertRecord(
                        expert_id=expert_id,
                        location=ExpertLocation.NVME,
                        payload=payload,
                    )
                )

                source_payloads[expert_id] = payload.clone()

            # One real Qwen-selected expert is enough to exercise
            # the full production lifecycle on every prediction.
            expert_id = predicted_ids[0]

            # Start every lifecycle case from cold RAM/GPU state.
            if ram.contains(expert_id):
                ram.remove(expert_id)

            if gpu.contains(expert_id):
                gpu.remove(expert_id)

            cancellation_case = (
                prediction_steps % 10 == 0
            )

            failure_case = (
                not cancellation_case
                and prediction_steps % 25 == 0
            )

            # ------------------------------------------------------
            # Cancellation:
            #
            # Stage 1 completes NVMe -> RAM, then cancellation is
            # observed at the stage boundary and Stage 2 is skipped.
            # ------------------------------------------------------
            cancellation = Event()

            def fallback_loader(
                task: PrefetchTask,
            ) -> None:
                raise AssertionError(
                    "fallback loader must not run"
                )

            if cancellation_case:
                cancellation_steps += 1

                real_nvme_to_ram = create_nvme_to_ram_handler(
                    nvme,
                    ram,
                )

                def cancel_after_stage_one(
                    task: PrefetchTask,
                ) -> None:
                    real_nvme_to_ram(task)
                    cancellation.set()

                transfer = create_storage_transfer_executor(
                    nvme,
                    ram,
                    fallback_loader,
                    gpu=gpu,
                    should_cancel=cancellation.is_set,
                    nvme_to_ram_handler=cancel_after_stage_one,
                )

            # ------------------------------------------------------
            # Failure:
            #
            # Inject one Stage-1 failure through the public factory
            # injection point. Stage 2 must never execute.
            # ------------------------------------------------------
            elif failure_case:
                failure_steps += 1

                def fail_stage_one(
                    task: PrefetchTask,
                ) -> None:
                    raise RuntimeError(
                        "injected R-5 NVMe -> RAM failure"
                    )

                transfer = create_storage_transfer_executor(
                    nvme,
                    ram,
                    fallback_loader,
                    gpu=gpu,
                    nvme_to_ram_handler=fail_stage_one,
                )

            # ------------------------------------------------------
            # Normal production path.
            # ------------------------------------------------------
            else:
                transfer = create_storage_transfer_executor(
                    nvme,
                    ram,
                    fallback_loader,
                    gpu=gpu,
                )

            engine = PrefetchEngine(
                loader=fallback_loader,
                num_workers=1,
                transfer_executor=transfer,
            )

            pipeline = PrefetchPipeline(
                scheduler,
                engine,
                auto_start=True,
            )

            try:
                # --------------------------------------------------
                # Pipeline must submit the selected real candidate.
                # --------------------------------------------------
                result = pipeline.process(current)

                submitted_ids = [
                    int(task.expert_id)
                    for task in result.submitted_tasks
                ]

                # A prediction can be filtered by admission because
                # another candidate is already resident. In that case
                # exercise the next selected candidate instead.
                if not submitted_ids:
                    pipeline.stop()
                    _assert_idle(engine)
                    continue

                assert expert_id in submitted_ids

                for submitted_id in submitted_ids:
                    _wait_terminal(
                        engine,
                        submitted_id,
                    )

                status = engine.task_status(expert_id)

                # --------------------------------------------------
                # Cancellation case.
                # --------------------------------------------------
                if cancellation_case:
                    assert status == PrefetchStatus.CANCELLED

                    cancelled_total += 1

                    # Stage 1 ran; Stage 2 did not.
                    assert ram.contains(expert_id)
                    assert gpu.get(expert_id) is None

                    assert expert_id in engine.cancelled_tasks
                    assert expert_id not in engine.completed_tasks
                    assert expert_id not in engine.failed_tasks

                    _assert_idle(engine)
                    continue

                # --------------------------------------------------
                # Failure case.
                # --------------------------------------------------
                if failure_case:
                    assert status == PrefetchStatus.FAILED

                    failed_total += 1

                    # Stage 1 failed before materializing RAM.
                    assert not ram.contains(expert_id)
                    assert gpu.get(expert_id) is None

                    assert expert_id in engine.failed_tasks
                    assert expert_id not in engine.completed_tasks
                    assert expert_id not in engine.cancelled_tasks

                    _assert_idle(engine)

                    # ----------------------------------------------
                    # Retry SAME Qwen-selected expert through the
                    # complete production transfer factory.
                    # ----------------------------------------------
                    retry_transfer = (
                        create_storage_transfer_executor(
                            nvme,
                            ram,
                            fallback_loader,
                            gpu=gpu,
                        )
                    )

                    retry_engine = PrefetchEngine(
                        loader=fallback_loader,
                        num_workers=1,
                        transfer_executor=retry_transfer,
                    )

                    retry_pipeline = PrefetchPipeline(
                        scheduler,
                        retry_engine,
                        auto_start=True,
                    )

                    try:
                        retry = retry_pipeline.process(current)

                        retry_ids = [
                            int(task.expert_id)
                            for task in retry.submitted_tasks
                        ]

                        assert retry_ids
                        assert expert_id in retry_ids

                        assert retry.admission_decisions
                        assert all(
                            decision.reason
                            != AdmissionReason.REJECT_DUPLICATE_QUEUED
                            for decision in retry.admission_decisions
                        )
                        assert all(
                            decision.reason
                            != AdmissionReason.REJECT_DUPLICATE_RUNNING
                            for decision in retry.admission_decisions
                        )

                        retry_submitted_total += 1

                        retry_status = _wait_terminal(
                            retry_engine,
                            expert_id,
                        )

                        assert (
                            retry_status
                            == PrefetchStatus.COMPLETED
                        )

                        retry_completed_total += 1

                        assert expert_id in (
                            retry_engine.completed_tasks
                        )

                        assert expert_id not in (
                            retry_engine.failed_tasks
                        )

                        assert expert_id not in (
                            retry_engine.cancelled_tasks
                        )

                        _assert_payload_integrity(
                            expert_id,
                            ram,
                            gpu,
                            source_payloads,
                        )

                        _assert_idle(retry_engine)

                    finally:
                        retry_pipeline.stop()

                    continue

                # --------------------------------------------------
                # Normal completion.
                # --------------------------------------------------
                assert status == PrefetchStatus.COMPLETED

                normal_completed += 1

                assert expert_id in engine.completed_tasks
                assert expert_id not in engine.failed_tasks
                assert expert_id not in engine.cancelled_tasks

                _assert_payload_integrity(
                    expert_id,
                    ram,
                    gpu,
                    source_payloads,
                )

                _assert_idle(engine)

            finally:
                pipeline.stop()

        torch.cuda.synchronize()

        # ----------------------------------------------------------
        # Final real Qwen health check.
        # ----------------------------------------------------------
        with torch.no_grad():
            outputs = model(
                input_ids=torch.tensor(
                    [[1]],
                    dtype=torch.long,
                    device="cuda",
                )
            )

        assert outputs.logits.shape[-1] == 128

        # ----------------------------------------------------------
        # Global gates.
        # ----------------------------------------------------------
        assert routing_steps == longrun_steps
        assert prediction_steps > 0

        assert normal_completed > 0

        assert cancellation_steps > 0
        assert cancelled_total == cancellation_steps

        assert failure_steps > 0
        assert failed_total == retry_completed_total
        assert retry_submitted_total == retry_completed_total

        print()
        print(
            "7-R-5 Real Qwen Production Long-Run"
        )
        print(
            "-----------------------------------"
        )
        print(
            f"warm-up steps:          {warmup_steps}"
        )
        print(
            f"routing steps:          {routing_steps}"
        )
        print(
            f"prediction steps:       {prediction_steps}"
        )
        print(
            f"normal completions:     {normal_completed}"
        )
        print(
            f"cancellation steps:     {cancellation_steps}"
        )
        print(
            f"cancelled tasks:        {cancelled_total}"
        )
        print(
            f"failure steps:          {failure_steps}"
        )
        print(
            f"failed first attempts:  {failed_total}"
        )
        print(
            f"retry submissions:      {retry_submitted_total}"
        )
        print(
            f"retry completions:      {retry_completed_total}"
        )
        print(
            f"NVMe experts:           {len(source_payloads)}"
        )
        print(
            "Qwen CUDA forward:      PASS"
        )
        print()
    finally:
        if root.exists():
            shutil.rmtree(root)

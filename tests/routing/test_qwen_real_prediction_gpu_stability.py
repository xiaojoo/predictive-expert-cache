from __future__ import annotations

import torch

from predictive_cache.cache import PredictiveExpertCache
from predictive_cache.prefetch import (
    PrefetchEngine,
    PrefetchPipeline,
    PrefetchSource,
    PrefetchTarget,
)
from predictive_cache.routing import (
    QwenMoeRoutingCapture,
    QwenRoutingBridge,
)
from predictive_cache.scheduler import ExpertScheduler

from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeForCausalLM,
)


def _build_real_qwen() -> Qwen3MoeForCausalLM:
    config = Qwen3MoeConfig(
        num_hidden_layers=2,
        hidden_size=128,
        intermediate_size=256,
        num_local_experts=8,
        num_experts_per_tok=2,
        vocab_size=128,
    )

    model = Qwen3MoeForCausalLM(config)
    model.eval()
    return model


def _capture_routing(
    model: Qwen3MoeForCausalLM,
    capture: QwenMoeRoutingCapture,
    input_ids: torch.Tensor,
    *,
    step: int,
) -> list[int]:
    with torch.no_grad():
        output = capture.capture_forward(
            model,
            input_ids=input_ids,
            step=step,
        )

    assert output.logits.shape[:2] == input_ids.shape

    events = capture.bridge.collector.events()
    assert events

    current_experts = sorted(
        {
            expert_id
            for event in events
            if event.step == step
            for expert_id in event.expert_ids
        }
    )

    assert current_experts
    return current_experts


def test_real_qwen_prediction_gpu_prefetch_multi_cycle_stability():
    """
    6E-7:

    Real Qwen routing
        ↓
    PredictiveExpertCache
        ↓
    prediction
        ↓
    ExpertScheduler
        ↓
    PrefetchPipeline
        ↓
    Admission
        ↓
    PrefetchEngine
        ↓
    NVME → RAM
        ↓
    RAM → GPU
        ↓
    GPU residency
        ↓
    repeated prediction cycles
        ↓
    resident experts suppressed
        ↓
    no duplicate prefetch
    """

    model = _build_real_qwen()
    cache = PredictiveExpertCache()

    bridge = QwenRoutingBridge(cache)
    capture = QwenMoeRoutingCapture(bridge)

    scheduler = ExpertScheduler(cache)

    input_ids = torch.tensor(
        [
            [10, 11, 12],
            [20, 21, 22],
        ],
        dtype=torch.long,
    )

    routing_history: list[list[int]] = []

    # ---------------------------------------------------------
    # Real Qwen routing history
    # ---------------------------------------------------------
    for step in range(4):
        current_experts = _capture_routing(
            model,
            capture,
            input_ids,
            step=step,
        )

        routing_history.append(current_experts)

    assert len(routing_history) == 4
    assert all(routing_history)

    # ---------------------------------------------------------
    # Use the same prediction context contract as 6E-6.
    # ---------------------------------------------------------
    prediction_context = [
        routing_history[-1][0],
    ]

    predictions = cache.predict(
        prediction_context,
        top_k=cache.config.prediction_top_k,
    )

    assert predictions

    predicted_ids = {
        prediction.expert_id
        for prediction in predictions
    }

    assert predicted_ids
    assert len(predicted_ids) <= cache.config.prediction_top_k
    assert len(predicted_ids) == len(set(predicted_ids))

    assert all(
        0 <= expert_id < model.config.num_local_experts
        for expert_id in predicted_ids
    )

    # ---------------------------------------------------------
    # The predicted experts must initially be non-resident.
    # ---------------------------------------------------------
    assert all(
        cache.get(expert_id) is None
        for expert_id in predicted_ids
    )

    # ---------------------------------------------------------
    # Physical storage simulation.
    # ---------------------------------------------------------
    ram_storage: dict[int, torch.Tensor] = {}
    gpu_storage: dict[int, torch.Tensor] = {}

    def nvme_to_ram(task):
        ram_storage[task.expert_id] = torch.empty(
            256,
            dtype=torch.float32,
        )

    def ram_to_gpu(task):
        tensor = ram_storage[task.expert_id]

        gpu_storage[task.expert_id] = tensor.to(
            "cuda",
            non_blocking=False,
        )

    cycle_records: list[dict[str, set[int]]] = []

    # =========================================================
    # Cycle 1
    # =========================================================
    nvme_engine = PrefetchEngine(
        nvme_to_ram,
        num_workers=1,
    )

    nvme_pipeline = PrefetchPipeline(
        scheduler=scheduler,
        engine=nvme_engine,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
    )

    try:
        first_result = nvme_pipeline.process(
            prediction_context,
        )

        first_scheduled_ids = {
            request.expert_id
            for request in first_result.scheduled_requests
        }

        first_submitted_ids = {
            task.expert_id
            for task in first_result.submitted_tasks
        }

        assert first_scheduled_ids == predicted_ids
        assert first_submitted_ids == predicted_ids

        assert all(
            decision.admitted
            for decision in first_result.admission_decisions
        )

        assert (
            first_result.admission_stats.accepted
            == len(predicted_ids)
        )

        nvme_pipeline.stop(wait=True)

        assert set(ram_storage) == predicted_ids

        # -----------------------------------------------------
        # RAM → GPU
        # -----------------------------------------------------
        for expert_id in predicted_ids:
            class TransferTask:
                pass

            task = TransferTask()
            task.expert_id = expert_id

            ram_to_gpu(task)

        assert set(gpu_storage) == predicted_ids

        assert all(
            tensor.is_cuda
            for tensor in gpu_storage.values()
        )

        # -----------------------------------------------------
        # Synchronize algorithm-level GPU residency.
        # -----------------------------------------------------
        for expert_id in predicted_ids:
            cache.insert(
                expert_id=expert_id,
                size_bytes=1024,
                location="gpu",
            )

        for expert_id in predicted_ids:
            entry = cache.get(expert_id)

            assert entry is not None
            assert entry.location == "gpu"

        cycle_records.append(
            {
                "predicted": set(predicted_ids),
                "scheduled": set(first_scheduled_ids),
                "resident": set(gpu_storage),
            }
        )

    finally:
        nvme_pipeline.stop(wait=True)

    # =========================================================
    # Cycle 2 + Cycle 3
    #
    # New physical pipelines are created because the underlying
    # PrefetchEngine is intentionally lifecycle-bound.
    # =========================================================
    for cycle in range(2, 4):
        cycle_engine = PrefetchEngine(
            nvme_to_ram,
            num_workers=1,
        )

        cycle_pipeline = PrefetchPipeline(
            scheduler=scheduler,
            engine=cycle_engine,
            source=PrefetchSource.NVME,
            target=PrefetchTarget.RAM,
        )

        try:
            cycle_predictions = cache.predict(
                prediction_context,
                top_k=cache.config.prediction_top_k,
            )

            assert cycle_predictions

            cycle_prediction_ids = {
                prediction.expert_id
                for prediction in cycle_predictions
            }

            cycle_result = cycle_pipeline.process(
                prediction_context,
            )

            cycle_scheduled_ids = {
                request.expert_id
                for request in cycle_result.scheduled_requests
            }

            # Already GPU-resident experts must never be scheduled
            # again as duplicate prefetches.
            assert cycle_scheduled_ids.isdisjoint(
                set(gpu_storage)
            )

            assert cycle_scheduled_ids.issubset(
                cycle_prediction_ids
            )

            # Since the first cycle materialized every first-cycle
            # prediction into GPU residency, a repeated prediction
            # of the same candidates must produce no duplicate work.
            if cycle_prediction_ids.issubset(
                set(gpu_storage)
            ):
                assert cycle_scheduled_ids == set()

            cycle_pipeline.stop(wait=True)

            # Physical GPU residency must remain unchanged.
            assert set(gpu_storage) == predicted_ids

            assert all(
                tensor.is_cuda
                for tensor in gpu_storage.values()
            )

            cycle_records.append(
                {
                    "predicted": set(cycle_prediction_ids),
                    "scheduled": set(cycle_scheduled_ids),
                    "resident": set(gpu_storage),
                }
            )

        finally:
            cycle_pipeline.stop(wait=True)

    # =========================================================
    # Cross-cycle stability invariants
    # =========================================================
    assert len(cycle_records) == 3

    assert cycle_records[0]["scheduled"] == predicted_ids
    assert cycle_records[0]["resident"] == predicted_ids

    for record in cycle_records:
        assert record["scheduled"].issubset(
            record["predicted"]
        )

        assert record["scheduled"].isdisjoint(
            record["resident"] - record["scheduled"]
        )

        assert record["resident"] == predicted_ids

    # No cycle after the initial materialization may introduce
    # another physical GPU copy.
    for record in cycle_records[1:]:
        assert record["scheduled"].isdisjoint(
            predicted_ids
        )

    # Final algorithm-level residency must still agree with the
    # physical GPU storage.
    for expert_id in predicted_ids:
        entry = cache.get(expert_id)

        assert entry is not None
        assert entry.location == "gpu"

        assert expert_id in gpu_storage
        assert gpu_storage[expert_id].is_cuda

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

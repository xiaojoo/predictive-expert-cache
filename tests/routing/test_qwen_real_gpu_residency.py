import torch

from predictive_cache.cache import PredictiveExpertCache
from predictive_cache.routing import (
    QwenMoeRoutingCapture,
    QwenRoutingBridge,
)
from predictive_cache.prefetch import (
    PrefetchEngine,
    PrefetchPipeline,
    PrefetchSource,
    PrefetchTarget,
)
from predictive_cache.prefetch.storage import (
    create_storage_transfer_executor,
)
from predictive_cache.scheduler import ExpertScheduler
from predictive_cache.storage.expert_record import ExpertRecord
from predictive_cache.storage.expert_store import ExpertLocation
from predictive_cache.storage.in_memory import InMemoryExpertStore
from predictive_cache.storage.ram import RAMExpertStore
from predictive_cache.storage.gpu import GPUExpertStore
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeForCausalLM,
)


def test_real_qwen_gpu_residency_suppresses_duplicate_prefetch():
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

    cache = PredictiveExpertCache()
    bridge = QwenRoutingBridge(cache)
    capture = QwenMoeRoutingCapture(bridge)

    input_ids = torch.tensor(
        [
            [10, 11, 12],
            [20, 21, 22],
        ],
        dtype=torch.long,
    )

    # =========================================================
    # Stage 1: Real Qwen routing
    # =========================================================

    with torch.no_grad():
        output = capture.capture_forward(
            model,
            input_ids=input_ids,
            step=0,
        )

    assert output.logits.shape[:2] == (2, 3)

    events = bridge.collector.events()
    assert events

    current_experts = sorted(
        {
            expert_id
            for event in events
            if event.step == 5
            for expert_id in event.expert_ids
        }
    )

    assert current_experts

    # =========================================================
    # Stage 2: Build storage tiers
    # =========================================================

    nvme = InMemoryExpertStore()
    ram = RAMExpertStore()
    gpu = GPUExpertStore()

    original_payloads = {}

    for expert_id in range(config.num_local_experts):
        payload = torch.tensor(
            [float(expert_id)],
            dtype=torch.float32,
        )

        original_payloads[expert_id] = payload.clone()

        nvme.put(
            ExpertRecord(
                expert_id=expert_id,
                location=ExpertLocation.NVME,
                payload=payload,
            )
        )

    # =========================================================
    # Stage 3: Real scheduler prediction
    # =========================================================

    scheduler = ExpertScheduler(cache)

    predictions = scheduler.plan_prefetch_predictions(
        current_experts
    )

    assert predictions

    predicted_experts = [
        request.expert_id
        for request in predictions
    ]

    assert predicted_experts

    assert set(predicted_experts).issubset(
        set(range(config.num_local_experts))
    )

    # Predictions should be unique after scheduler processing.
    assert len(predicted_experts) == len(
        set(predicted_experts)
    )

    # =========================================================
    # Stage 4: NVME -> RAM -> GPU
    # =========================================================

    nvme_to_ram_executor = create_storage_transfer_executor(
        nvme,
        ram,
        lambda task: None,
        gpu=gpu,
    )

    nvme_engine = PrefetchEngine(
        lambda task: None,
        num_workers=1,
        transfer_executor=nvme_to_ram_executor,
    )

    nvme_pipeline = PrefetchPipeline(
        scheduler=scheduler,
        engine=nvme_engine,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
    )

    try:
        ram_result = nvme_pipeline.process(
            current_experts
        )

        assert ram_result.submitted_tasks

    finally:
        nvme_pipeline.stop()

    ram_experts = [
        task.expert_id
        for task in ram_result.submitted_tasks
    ]

    assert ram_experts

    ram_set = set(ram_experts)

    assert ram_set.issubset(
        set(range(config.num_local_experts))
    )

    # Validate actual RAM residency.
    for expert_id in ram_experts:
        record = ram.get(expert_id)

        assert record is not None
        assert record.location == ExpertLocation.RAM
        assert record.payload.device.type == "cpu"

        assert torch.equal(
            record.payload,
            original_payloads[expert_id],
        )

    # RAM -> GPU
    gpu_executor = create_storage_transfer_executor(
        nvme,
        ram,
        lambda task: None,
        gpu=gpu,
    )

    gpu_engine = PrefetchEngine(
        lambda task: None,
        num_workers=1,
        transfer_executor=gpu_executor,
    )

    # Explicit scheduler is used here because Stage 1 has already
    # determined exactly which predicted experts reached RAM.
    class _ResidentScheduler:
        def __init__(self, expert_ids):
            self._expert_ids = list(expert_ids)

        def plan_prefetch_predictions(
            self,
            current_experts,
        ):
            return [
                type(
                    "PrefetchRequest",
                    (),
                    {
                        "expert_id": expert_id,
                        "priority": 1.0,
                        "confidence": 1.0,
                        "estimated_distance": 0,
                    },
                )()
                for expert_id in self._expert_ids
            ]

    gpu_pipeline = PrefetchPipeline(
        scheduler=_ResidentScheduler(ram_experts),
        engine=gpu_engine,
        source=PrefetchSource.RAM,
        target=PrefetchTarget.GPU,
    )

    try:
        gpu_result = gpu_pipeline.process(
            ram_experts
        )

        assert gpu_result.submitted_tasks

    finally:
        gpu_pipeline.stop()

    gpu_experts = {
        task.expert_id
        for task in gpu_result.submitted_tasks
    }

    assert gpu_experts == ram_set

    # =========================================================
    # Stage 5: Verify actual GPU residency
    # =========================================================

    for expert_id in ram_experts:
        record = gpu.get(expert_id)

        assert record is not None
        assert record.location == ExpertLocation.GPU
        assert record.payload.device.type == "cuda"

        assert torch.equal(
            record.payload.cpu(),
            original_payloads[expert_id],
        )

    # =========================================================
    # Stage 6: Feed GPU residency back into scheduler
    #
    # This is the important 6D-5 assertion:
    # already resident GPU experts must not be scheduled again.
    # =========================================================

    resident = set(
        gpu.get(expert_id).expert_id
        for expert_id in ram_experts
        if gpu.get(expert_id) is not None
    )

    assert resident == ram_set

    second_predictions = scheduler.plan_predictions(
        cache.predict(
            current_experts,
            top_k=config.num_local_experts,
        ),
        cached_experts=resident,
    )

    second_predicted_experts = {
        request.expert_id
        for request in second_predictions
    }

    assert second_predicted_experts.isdisjoint(
        resident
    )

    # =========================================================
    # Stage 7: Explicitly verify each resident expert is skipped
    # =========================================================

    for expert_id in resident:
        repeated_predictions = scheduler.plan_predictions(
            cache.predict(
                [expert_id],
                top_k=config.num_local_experts,
            ),
            cached_experts={expert_id},
        )

        assert expert_id not in {
            request.expert_id
            for request in repeated_predictions
        }

    # =========================================================
    # Stage 8: GPU payload integrity
    # =========================================================

    for expert_id in resident:
        record = gpu.get(expert_id)

        assert record is not None
        assert record.location == ExpertLocation.GPU
        assert record.payload.device.type == "cuda"

        assert torch.equal(
            record.payload.cpu(),
            original_payloads[expert_id],
        )
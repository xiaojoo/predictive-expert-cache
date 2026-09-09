import torch

from predictive_cache.cache import PredictiveExpertCache
from predictive_cache.routing import QwenMoeRoutingCapture
from predictive_cache.routing import QwenRoutingBridge
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


class _ExplicitScheduler:
    def __init__(self, expert_ids):
        self._expert_ids = list(expert_ids)

    def plan_prefetch_predictions(self, current_experts):
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


def test_real_qwen_nvme_ram_gpu():
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

    scheduler = ExpertScheduler(cache)

    nvme = InMemoryExpertStore()
    ram = RAMExpertStore()
    gpu = GPUExpertStore()

    for expert_id in range(config.num_local_experts):
        nvme.put(
            ExpertRecord(
                expert_id=expert_id,
                location=ExpertLocation.NVME,
                payload=torch.tensor(
                    [float(expert_id)],
                    dtype=torch.float32,
                ),
            )
        )

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
        ram_result = nvme_pipeline.process(current_experts)

        assert ram_result.submitted_tasks

    finally:
        nvme_pipeline.stop()

    ram_experts = [
        task.expert_id
        for task in ram_result.submitted_tasks
    ]

    assert ram_experts

    for expert_id in ram_experts:
        record = ram.get(expert_id)

        assert record is not None
        assert record.location == ExpertLocation.RAM
        assert record.payload.device.type == "cpu"

    ram_to_gpu_executor = create_storage_transfer_executor(
        nvme,
        ram,
        lambda task: None,
        gpu=gpu,
    )

    gpu_engine = PrefetchEngine(
        lambda task: None,
        num_workers=1,
        transfer_executor=ram_to_gpu_executor,
    )

    gpu_scheduler = _ExplicitScheduler(ram_experts)

    gpu_pipeline = PrefetchPipeline(
        scheduler=gpu_scheduler,
        engine=gpu_engine,
        source=PrefetchSource.RAM,
        target=PrefetchTarget.GPU,
    )

    try:
        gpu_result = gpu_pipeline.process(ram_experts)

        assert gpu_result.submitted_tasks

        assert {
            task.expert_id
            for task in gpu_result.submitted_tasks
        } == set(ram_experts)

    finally:
        gpu_pipeline.stop()

    for expert_id in ram_experts:
        record = gpu.get(expert_id)

        assert record is not None
        assert record.location == ExpertLocation.GPU
        assert record.payload.device.type == "cuda"

        assert torch.equal(
            record.payload.cpu(),
            torch.tensor(
                [float(expert_id)],
                dtype=torch.float32,
            ),
        )

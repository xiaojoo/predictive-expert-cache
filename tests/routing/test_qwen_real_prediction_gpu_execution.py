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


def _build_real_qwen():
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

    return model, config


def _run_real_qwen_routing(model, bridge):
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

    return current_experts


def _build_gpu_transfer_executor(ram, gpu):
    """
    Build the real RAM -> GPU transfer executor using the same
    transfer abstraction used by the existing GPU integration path.

    The storage.py factory intentionally only registers NVMe -> RAM,
    so RAM -> GPU must be registered explicitly here.
    """

    from predictive_cache.prefetch.transfer import (
        PrefetchTransferExecutor,
    )
    from predictive_cache.prefetch.types import (
        PrefetchSource,
        PrefetchTarget,
    )

    transfer = PrefetchTransferExecutor(
        lambda task: None,
    )

    def ram_to_gpu(task):
        record = ram.get(task.expert_id)

        if record is None:
            raise KeyError(
                f"expert {task.expert_id!r} is not present in RAM"
            )

        payload = record.payload

        if not isinstance(payload, torch.Tensor):
            raise TypeError(
                "RAM payload must be a torch.Tensor"
            )

        gpu.put(
            ExpertRecord(
                expert_id=task.expert_id,
                location=ExpertLocation.GPU,
                payload=payload.to("cuda"),
            )
        )

    transfer.register(
        PrefetchSource.RAM,
        PrefetchTarget.GPU,
        ram_to_gpu,
    )

    return transfer


def test_real_qwen_prediction_drives_gpu_prefetch_execution():
    """
    6E-5

    Real Qwen routing
        ↓
    PredictiveExpertCache
        ↓
    prediction
        ↓
    ExpertScheduler
        ↓
    PrefetchRequest
        ↓
    PrefetchPipeline
        ↓
    Admission
        ↓
    PrefetchEngine
        ↓
    NVMe → RAM
        ↓
    RAM → GPU
        ↓
    GPU resident

    The test validates actual asynchronous PrefetchEngine execution,
    not merely task submission.
    """

    if not torch.cuda.is_available():
        raise RuntimeError(
            "6E-5 requires CUDA for real GPU execution"
        )

    model, config = _build_real_qwen()

    cache = PredictiveExpertCache()
    bridge = QwenRoutingBridge(cache)

    current_experts = _run_real_qwen_routing(
        model,
        bridge,
    )

    scheduler = ExpertScheduler(cache)

    prediction_context = [current_experts[0]]

    predictions = cache.predict(
        prediction_context,
        top_k=4,
    )

    assert predictions

    predicted_ids = {
        prediction.expert_id
        for prediction in predictions
    }

    assert predicted_ids

    assert all(
        0 <= expert_id < config.num_local_experts
        for expert_id in predicted_ids
    )

    # ---------------------------------------------------------
    # NVMe source
    # ---------------------------------------------------------

    nvme = InMemoryExpertStore()
    ram = RAMExpertStore()
    gpu = GPUExpertStore()

    payloads = {}

    for expert_id in predicted_ids:
        payload = torch.tensor(
            [
                float(expert_id),
                float(expert_id) + 0.5,
            ],
            dtype=torch.float32,
        )

        payloads[expert_id] = payload.clone()

        nvme.put(
            ExpertRecord(
                expert_id=expert_id,
                location=ExpertLocation.NVME,
                payload=payload,
            )
        )

    # ---------------------------------------------------------
    # Prediction-driven NVMe -> RAM
    # ---------------------------------------------------------

    nvme_to_ram_executor = create_storage_transfer_executor(
        nvme,
        ram,
        lambda task: None,
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

    ram_result = nvme_pipeline.process(
        prediction_context,
    )

    assert ram_result.scheduled_requests
    assert ram_result.submitted_tasks

    scheduled_ids = {
        request.expert_id
        for request in ram_result.scheduled_requests
    }

    submitted_ids = {
        task.expert_id
        for task in ram_result.submitted_tasks
    }

    assert scheduled_ids.issubset(predicted_ids)
    assert submitted_ids == scheduled_ids

    # process() is asynchronous. Drain the engine before checking
    # completed_tasks / results.
    nvme_pipeline.stop(wait=True)

    assert set(nvme_engine.completed_tasks) == submitted_ids
    assert not nvme_engine.failed_tasks
    assert nvme_engine.unfinished_tasks == 0

    completed_results = {
        result.expert_id
        for result in nvme_engine.results()
        if result.status.value == "completed"
    }

    assert completed_results == submitted_ids

    # ---------------------------------------------------------
    # Verify actual RAM residency.
    # ---------------------------------------------------------

    for expert_id in submitted_ids:
        record = ram.get(expert_id)

        assert record is not None
        assert record.location == ExpertLocation.RAM
        assert record.payload.device.type == "cpu"

        assert torch.equal(
            record.payload,
            payloads[expert_id],
        )

    # ---------------------------------------------------------
    # RAM -> GPU
    #
    # Use the actual submitted prediction-driven request set.
    # No new expert candidates are introduced here.
    # ---------------------------------------------------------

    gpu_transfer_executor = _build_gpu_transfer_executor(
        ram,
        gpu,
    )

    gpu_engine = PrefetchEngine(
        lambda task: None,
        num_workers=1,
        transfer_executor=gpu_transfer_executor,
    )

    class _CommittedPredictionScheduler:
        """
        Second-stage scheduler.

        It does not generate new predictions. It only carries the
        exact requests that survived prediction + scheduler +
        admission in the first stage into the already established
        RAM -> GPU execution stage.
        """

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
                        "score": 1.0,
                    },
                )()
                for expert_id in self._expert_ids
            ]

    gpu_scheduler = _CommittedPredictionScheduler(
        sorted(submitted_ids),
    )

    gpu_pipeline = PrefetchPipeline(
        scheduler=gpu_scheduler,
        engine=gpu_engine,
        source=PrefetchSource.RAM,
        target=PrefetchTarget.GPU,
    )

    gpu_result = gpu_pipeline.process(
        sorted(submitted_ids),
    )

    assert gpu_result.submitted_tasks

    gpu_submitted_ids = {
        task.expert_id
        for task in gpu_result.submitted_tasks
    }

    assert gpu_submitted_ids == submitted_ids

    gpu_pipeline.stop(wait=True)

    assert set(gpu_engine.completed_tasks) == submitted_ids
    assert not gpu_engine.failed_tasks
    assert gpu_engine.unfinished_tasks == 0

    gpu_completed_results = {
        result.expert_id
        for result in gpu_engine.results()
        if result.status.value == "completed"
    }

    assert gpu_completed_results == submitted_ids

    # ---------------------------------------------------------
    # Verify actual GPU residency.
    # ---------------------------------------------------------

    for expert_id in submitted_ids:
        record = gpu.get(expert_id)

        assert record is not None
        assert record.location == ExpertLocation.GPU
        assert record.payload.device.type == "cuda"

        assert torch.equal(
            record.payload.cpu(),
            payloads[expert_id],
        )

    # ---------------------------------------------------------
    # Final 6E-5 closure
    # ---------------------------------------------------------

    assert submitted_ids.issubset(predicted_ids)

    assert {
        expert_id
        for expert_id in submitted_ids
        if gpu.contains(expert_id)
    } == submitted_ids
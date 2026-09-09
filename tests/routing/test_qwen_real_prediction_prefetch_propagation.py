from __future__ import annotations

import torch

from predictive_cache.cache import PredictiveExpertCache
from predictive_cache.prefetch import (
    PrefetchEngine,
    PrefetchPipeline,
    PrefetchSource,
    PrefetchTarget,
)
from predictive_cache.routing import QwenMoeRoutingCapture
from predictive_cache.routing import QwenRoutingBridge
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


def test_real_qwen_prediction_propagates_to_prefetch_pipeline():
    """
    6E-3:

    Real Qwen routing
        -> PredictiveExpertCache
        -> prediction
        -> ExpertScheduler
        -> PrefetchPipeline
        -> scheduled_requests
        -> submitted_tasks
        -> admission
    """

    model = _build_real_qwen()

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

    routing_history: list[list[int]] = []

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

    # Keep the prediction context intentionally narrow so that historical
    # experts remain available as prediction candidates.
    prediction_context = [routing_history[-1][0]]

    predictions = cache.predict(
        prediction_context,
        top_k=cache.config.prediction_top_k,
    )

    assert predictions

    predicted_ids = [
        prediction.expert_id
        for prediction in predictions
    ]

    assert predicted_ids
    assert len(predicted_ids) <= cache.config.prediction_top_k
    assert len(predicted_ids) == len(set(predicted_ids))

    assert all(
        0 <= expert_id < model.config.num_local_experts
        for expert_id in predicted_ids
    )

    # Predictions are not resident in the metadata cache, therefore they
    # remain eligible for prefetch.
    assert all(
        cache.get(expert_id) is None
        for expert_id in predicted_ids
    )

    scheduler = ExpertScheduler(cache)

    scheduled_requests = scheduler.plan_prefetch(
        prediction_context
    )

    scheduled_ids = [
        request.expert_id
        for request in scheduled_requests
    ]

    assert scheduled_ids == predicted_ids

    for request, prediction in zip(
        scheduled_requests,
        predictions,
        strict=True,
    ):
        assert request.expert_id == prediction.expert_id
        assert request.score == prediction.score
        assert request.priority == prediction.score

    # Real PrefetchPipeline.
    # The engine callback intentionally performs no storage operation.
    # Storage transfer is already covered by 6D.
    engine = PrefetchEngine(
        lambda task: None,
        num_workers=1,
    )

    pipeline = PrefetchPipeline(
        scheduler=scheduler,
        engine=engine,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
    )

    try:
        result = pipeline.process(prediction_context)

        assert result.current_experts == prediction_context

        # Scheduler -> Pipeline.
        assert [
            request.expert_id
            for request in result.scheduled_requests
        ] == predicted_ids

        # Pipeline -> Engine.
        assert [
            task.expert_id
            for task in result.submitted_tasks
        ] == predicted_ids

        # Request -> Task field propagation.
        for request, task in zip(
            result.scheduled_requests,
            result.submitted_tasks,
            strict=True,
        ):
            assert task.expert_id == request.expert_id
            assert task.priority == request.priority
            assert task.confidence == request.score
            assert task.source == PrefetchSource.NVME
            assert task.target == PrefetchTarget.RAM

        # Admission decisions.
        assert len(result.admission_decisions) == len(
            predicted_ids
        )

        assert all(
            decision.admitted
            for decision in result.admission_decisions
        )

        # Use the actual AdmissionStats API.
        admission_stats = result.admission_stats

        assert admission_stats.accepted == len(predicted_ids)
        assert admission_stats.rejected == 0

        # Final propagation invariants.
        assert set(predicted_ids) == {
            request.expert_id
            for request in result.scheduled_requests
        }

        assert set(predicted_ids) == {
            task.expert_id
            for task in result.submitted_tasks
        }

    finally:
        pipeline.stop()
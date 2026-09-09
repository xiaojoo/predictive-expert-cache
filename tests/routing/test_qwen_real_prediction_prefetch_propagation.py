import torch

from predictive_cache.cache import PredictiveExpertCache
from predictive_cache.prefetch.engine import PrefetchEngine
from predictive_cache.prefetch.pipeline import (
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
        outputs = capture.capture_forward(
            model,
            input_ids=input_ids,
            step=step,
        )

    assert outputs.logits.shape[:2] == input_ids.shape

    events = capture.bridge.collector.events()

    step_events = [
        event
        for event in events
        if event.step == step
    ]

    assert step_events

    expert_ids: set[int] = set()

    for event in step_events:
        expert_ids.update(event.expert_ids)

    result = sorted(expert_ids)

    assert result
    assert all(
        0 <= expert_id < model.config.num_local_experts
        for expert_id in result
    )

    return result


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

    # Scheduler output is a valid candidate set derived from predictions.
    # The scheduler is allowed to apply its canonical ordering, so the
    # prediction list order is not part of the contract.
    assert set(scheduled_ids) == set(predicted_ids)
    assert len(scheduled_ids) == len(predicted_ids)

    prediction_map = {
        prediction.expert_id: prediction
        for prediction in predictions
    }

    for request in scheduled_requests:
        prediction = prediction_map[request.expert_id]

        assert request.expert_id in prediction_map
        assert request.score == prediction.score
        assert request.priority == prediction.score

    # Real PrefetchPipeline.
    #
    # The engine callback intentionally performs no storage operation.
    # Actual NVME -> RAM and RAM -> GPU transfer behavior is covered by
    # the 6D storage integration tests.
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
        #
        # Pipeline must preserve the predicted candidate set, but it is
        # allowed to receive the scheduler's canonical ordering.
        pipeline_scheduled_ids = [
            request.expert_id
            for request in result.scheduled_requests
        ]

        assert set(pipeline_scheduled_ids) == set(predicted_ids)
        assert len(pipeline_scheduled_ids) == len(predicted_ids)

        pipeline_prediction_map = {
            prediction.expert_id: prediction
            for prediction in predictions
        }

        for request in result.scheduled_requests:
            prediction = pipeline_prediction_map[request.expert_id]

            assert request.expert_id in pipeline_prediction_map
            assert request.score == prediction.score
            assert request.confidence == prediction.score
            assert (
                request.estimated_distance
                == prediction.estimated_distance
            )
            assert request.priority >= 0.0

        # Pipeline -> submitted tasks.
        submitted_ids = [
            task.expert_id
            for task in result.submitted_tasks
        ]

        assert set(submitted_ids) == set(predicted_ids)
        assert len(submitted_ids) == len(predicted_ids)

        # Admission must have evaluated every scheduled request.
        assert len(result.admission_decisions) == len(
            result.scheduled_requests
        )

        # All requests are accepted by the default admission configuration
        # used in this integration test.
        assert result.admission_stats.accepted == len(
            result.scheduled_requests
        )
        assert result.admission_stats.rejected == 0

        for decision in result.admission_decisions:
            assert decision.admitted is True

    finally:
        pipeline.stop(wait=True)
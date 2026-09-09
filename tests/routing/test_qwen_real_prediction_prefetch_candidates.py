from __future__ import annotations

import torch

from predictive_cache.cache import PredictiveExpertCache
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


def test_real_qwen_prediction_drives_prefetch_candidates_and_filters_resident():
    """
    6E-4:

    Real Qwen routing
        -> PredictiveExpertCache
        -> prediction
        -> prefetch_candidates
        -> ExpertScheduler
        -> PrefetchRequest

    Correctness invariants:

    1. Predicted non-resident experts become prefetch candidates.
    2. A predicted expert that is already resident is suppressed.
    3. An unrelated non-predicted resident expert does not appear in
       scheduler output.
    4. Scheduler output is exactly the non-resident predicted set.
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

    # As in 6E-1/6E-2, use a deliberately narrow prediction context so
    # historical experts remain available as candidates.
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

    # ---------------------------------------------------------
    # Phase 1:
    # Every prediction is initially non-resident.
    # Therefore all predictions must be prefetch candidates.
    # ---------------------------------------------------------

    candidates = cache.prefetch_candidates(
        prediction_context
    )

    candidate_ids = [
        prediction.expert_id
        for prediction in candidates
    ]

    assert candidate_ids == predicted_ids

    assert all(
        not cache.contains(expert_id)
        for expert_id in predicted_ids
    )

    # ---------------------------------------------------------
    # Phase 2:
    # The first predicted expert becomes resident.
    #
    # The predictor output itself remains unchanged, but the
    # prefetch candidate set must remove the resident expert.
    # ---------------------------------------------------------

    resident_predicted = predicted_ids[0]

    cache.insert(
        resident_predicted,
        size_bytes=1024,
        location="gpu",
    )

    assert cache.contains(resident_predicted)

    predictions_after_residency = cache.predict(
        prediction_context,
        top_k=cache.config.prediction_top_k,
    )

    prediction_ids_after_residency = [
        prediction.expert_id
        for prediction in predictions_after_residency
    ]

    # Residency must not alter the prediction layer.
    assert prediction_ids_after_residency == predicted_ids

    candidates_after_residency = cache.prefetch_candidates(
        prediction_context
    )

    candidate_ids_after_residency = [
        prediction.expert_id
        for prediction in candidates_after_residency
    ]

    expected_non_resident = [
        expert_id
        for expert_id in predicted_ids
        if expert_id != resident_predicted
    ]

    assert candidate_ids_after_residency == expected_non_resident

    assert resident_predicted not in candidate_ids_after_residency

    # ---------------------------------------------------------
    # Phase 3:
    # Insert an unrelated expert that was not predicted.
    #
    # It must remain irrelevant to prediction-driven prefetch.
    # ---------------------------------------------------------

    all_experts = set(range(model.config.num_local_experts))

    non_predicted_ids = sorted(
        all_experts - set(predicted_ids)
    )

    assert non_predicted_ids

    unrelated_resident = non_predicted_ids[0]

    cache.insert(
        unrelated_resident,
        size_bytes=1024,
        location="gpu",
    )

    assert cache.contains(unrelated_resident)

    final_candidates = cache.prefetch_candidates(
        prediction_context
    )

    final_candidate_ids = [
        prediction.expert_id
        for prediction in final_candidates
    ]

    assert final_candidate_ids == expected_non_resident

    assert unrelated_resident not in final_candidate_ids

    # ---------------------------------------------------------
    # Phase 4:
    # Validate the same filtering through the real Scheduler.
    # ---------------------------------------------------------

    scheduler = ExpertScheduler(cache)

    requests = scheduler.plan_prefetch_predictions(
        prediction_context
    )

    request_ids = [
        request.expert_id
        for request in requests
    ]

    assert set(request_ids) == set(expected_non_resident)

    assert resident_predicted not in request_ids
    assert unrelated_resident not in request_ids

    # No duplicate prefetch requests.
    assert len(request_ids) == len(set(request_ids))

    # Scheduler must not invent experts outside the prediction set.
    assert set(request_ids).issubset(
        set(predicted_ids)
    )

    # Every surviving request must correspond to a prediction.
    prediction_map = {
        prediction.expert_id: prediction
        for prediction in predictions
    }

    assert set(request_ids).issubset(
        set(prediction_map)
    )

    # The scheduler is allowed to reject a prediction when its
    # economic benefit is not positive enough. Therefore the
    # scheduler output must be a subset of the predicted set,
    # rather than an exact copy of it.
    #
    # The important correctness boundary is:
    #
    # prediction
    #     ↓
    # optional economic filtering
    #     ↓
    # PrefetchRequest
    #
    # but never:
    #
    # non-predicted expert
    #     ↓
    # PrefetchRequest
    assert set(request_ids).issubset(
        set(predicted_ids)
    )

    # Resident predicted experts must always be suppressed,
    # regardless of their prediction score.
    assert resident_predicted not in request_ids

    # The unrelated resident expert was never predicted and
    # therefore must never appear in scheduler output.
    assert unrelated_resident not in request_ids

    # Every surviving request must be genuinely non-resident.
    assert all(
        not cache.contains(expert_id)
        for expert_id in request_ids
    )

    # The surviving request's confidence must still correspond
    # exactly to the original predictor score.
    for request in requests:
        prediction = prediction_map[request.expert_id]

        assert request.expert_id == prediction.expert_id
        assert request.confidence == prediction.score
        assert request.confidence > 0.0
        assert request.priority > 0.0

    # At least one non-resident prediction should survive this
    # real-Qwen scheduling path in this test.
    assert request_ids

    # Final correctness invariant:
    #
    # scheduler output
    #   ⊆
    # predicted experts
    #
    # while resident and unrelated experts are excluded.
    assert set(request_ids).issubset(
        set(predicted_ids)
    )
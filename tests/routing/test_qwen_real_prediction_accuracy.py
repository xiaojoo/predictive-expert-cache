import torch

from predictive_cache.cache import PredictiveExpertCache
from predictive_cache.routing import (
    QwenMoeRoutingCapture,
    QwenRoutingBridge,
)
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


def _run_real_qwen(
    model,
    cache,
    *,
    input_ids,
    step,
):
    bridge = QwenRoutingBridge(cache)
    capture = QwenMoeRoutingCapture(bridge)

    with torch.no_grad():
        output = capture.capture_forward(
            model,
            input_ids=input_ids,
            step=step,
        )

    assert output.logits.shape[:2] == input_ids.shape

    events = bridge.collector.events()

    assert events

    routed_experts = sorted(
        {
            expert_id
            for event in events
            for expert_id in event.expert_ids
        }
    )

    assert routed_experts

    return events, routed_experts


def _prediction_metrics(
    predicted_experts,
    actual_experts,
):
    predicted = set(predicted_experts)
    actual = set(actual_experts)

    overlap = predicted & actual

    precision = (
        len(overlap) / len(predicted)
        if predicted
        else 0.0
    )

    recall = (
        len(overlap) / len(actual)
        if actual
        else 0.0
    )

    overlap_ratio = (
        len(overlap) / len(predicted | actual)
        if predicted | actual
        else 0.0
    )

    return {
        "predicted_count": len(predicted),
        "actual_count": len(actual),
        "overlap_count": len(overlap),
        "precision": precision,
        "recall": recall,
        "overlap_ratio": overlap_ratio,
    }


def test_real_qwen_prediction_accuracy_and_overlap():
    """
    Validate prediction accuracy against a subsequent real
    Qwen routing event.

    This test intentionally validates metric correctness rather
    than requiring a positive overlap from a randomly initialized
    Qwen MoE model.

    The lifecycle is:

        1. Real Qwen routing history
        2. Predictor learning
        3. Prediction generation
        4. Next real Qwen routing
        5. Prediction/actual intersection
        6. Precision / recall / overlap validation
    """

    model, config = _build_real_qwen()

    cache = PredictiveExpertCache()

    input_ids = torch.tensor(
        [
            [10, 11, 12],
            [20, 21, 22],
        ],
        dtype=torch.long,
    )

    valid_experts = set(
        range(config.num_local_experts)
    )

    # =========================================================
    # Stage 1: Build real Qwen routing history
    # =========================================================

    routing_history = []

    for step in range(4):
        events, experts = _run_real_qwen(
            model,
            cache,
            input_ids=input_ids,
            step=step,
        )

        assert events
        assert experts

        assert set(experts).issubset(
            valid_experts
        )

        routing_history.append(experts)

    assert len(routing_history) == 4

    # Predictor must contain real routing history.
    assert cache.predictor.step >= 4
    assert cache.predictor.stats
    assert cache.predictor.transitions

    # =========================================================
    # Stage 2: Establish prediction context
    # =========================================================

    historical_experts = sorted(
        {
            expert_id
            for experts in routing_history
            for expert_id in experts
        }
    )

    assert len(historical_experts) >= 2

    # Use only one current expert as prediction context.
    #
    # ExpertPredictor deliberately excludes current experts
    # from candidate predictions. Passing the entire Qwen
    # routing set can therefore legitimately remove every
    # candidate.
    prediction_context = [
        routing_history[-1][0]
    ]

    assert set(prediction_context).issubset(
        valid_experts
    )

    # At least one historical candidate must remain.
    assert any(
        expert_id not in prediction_context
        for expert_id in historical_experts
    )

    # =========================================================
    # Stage 3: Generate prediction
    # =========================================================

    predictions = cache.predict(
        current_experts=prediction_context,
        top_k=config.num_experts_per_tok,
    )

    assert predictions

    predicted_experts = [
        prediction.expert_id
        for prediction in predictions
    ]

    assert predicted_experts

    assert len(predicted_experts) == len(
        set(predicted_experts)
    )

    assert set(predicted_experts).issubset(
        valid_experts
    )

    # Current experts must be excluded from predictions.
    assert not (
        set(predicted_experts)
        & set(prediction_context)
    )

    # Predictions must be deterministically ranked.
    scores = [
        prediction.score
        for prediction in predictions
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )

    # Every prediction must contain valid components.
    for prediction in predictions:
        assert prediction.score >= 0.0
        assert prediction.frequency_score >= 0.0
        assert prediction.recency_score >= 0.0
        assert prediction.transition_score >= 0.0

        if prediction.distance is not None:
            assert prediction.distance >= 0.0

        if prediction.estimated_distance is not None:
            assert prediction.estimated_distance >= 0.0

    # =========================================================
    # Stage 4: Execute the next real Qwen routing event
    # =========================================================

    next_events, actual_experts = _run_real_qwen(
        model,
        cache,
        input_ids=input_ids,
        step=4,
    )

    assert next_events
    assert actual_experts

    assert set(actual_experts).issubset(
        valid_experts
    )

    # =========================================================
    # Stage 5: Calculate prediction accuracy
    # =========================================================

    metrics = _prediction_metrics(
        predicted_experts,
        actual_experts,
    )

    predicted_set = set(predicted_experts)
    actual_set = set(actual_experts)
    overlap_set = (
        predicted_set & actual_set
    )

    # Counts must exactly match the underlying sets.
    assert metrics["predicted_count"] == len(
        predicted_set
    )

    assert metrics["actual_count"] == len(
        actual_set
    )

    assert metrics["overlap_count"] == len(
        overlap_set
    )

    # =========================================================
    # Stage 6: Validate precision
    # =========================================================

    expected_precision = (
        len(overlap_set)
        / len(predicted_set)
    )

    assert (
        metrics["precision"]
        == expected_precision
    )

    assert 0.0 <= metrics["precision"] <= 1.0

    # =========================================================
    # Stage 7: Validate recall
    # =========================================================

    expected_recall = (
        len(overlap_set)
        / len(actual_set)
    )

    assert (
        metrics["recall"]
        == expected_recall
    )

    assert 0.0 <= metrics["recall"] <= 1.0

    # =========================================================
    # Stage 8: Validate Jaccard-style overlap ratio
    # =========================================================

    union_size = len(
        predicted_set | actual_set
    )

    expected_overlap_ratio = (
        len(overlap_set) / union_size
    )

    assert (
        metrics["overlap_ratio"]
        == expected_overlap_ratio
    )

    assert 0.0 <= metrics["overlap_ratio"] <= 1.0

    # =========================================================
    # Stage 9: Cross-check metric relationships
    # =========================================================

    assert (
        metrics["overlap_count"]
        <= metrics["predicted_count"]
    )

    assert (
        metrics["overlap_count"]
        <= metrics["actual_count"]
    )

    if metrics["overlap_count"] == 0:
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["overlap_ratio"] == 0.0

    if metrics["overlap_count"] > 0:
        assert metrics["precision"] > 0.0
        assert metrics["recall"] > 0.0
        assert metrics["overlap_ratio"] > 0.0

    # =========================================================
    # Stage 10: Verify top-k contract
    # =========================================================

    assert len(predictions) <= (
        config.num_experts_per_tok
    )

    assert len(predicted_experts) <= (
        config.num_experts_per_tok
    )

    # =========================================================
    # Stage 11: Ensure actual routing is independent
    # =========================================================

    # Prediction accuracy is measured against the real next
    # routing event. We intentionally do not mutate the cache
    # with predicted experts here, because doing so would turn
    # this into a cache-residency test rather than an accuracy
    # test.
    assert actual_experts

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


def _lookup_experts(
    cache,
    experts,
):
    hits = []
    misses = []

    for expert_id in experts:
        entry = cache.get(expert_id)

        if entry is None:
            misses.append(expert_id)
        else:
            hits.append(expert_id)

    return hits, misses


def test_real_qwen_multistep_prediction_produces_cache_hits():
    """
    Validate the real multi-step prediction/cache-hit lifecycle.

    The test deliberately separates:

        1. Real Qwen routing observations
        2. Predictor learning
        3. Prediction generation
        4. Prediction materialization
        5. Cache hit accounting
        6. A subsequent real Qwen routing lookup

    Prediction accuracy against the next real Qwen routing event
    is intentionally not asserted here. That belongs to the
    subsequent prediction-overlap phase.
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

    # Real routing must have advanced the predictor.
    assert cache.predictor.step >= 4

    # Predictor must contain real expert statistics.
    assert cache.predictor.stats

    # Multiple routing events must have produced
    # transition information.
    assert cache.predictor.transitions

    # =========================================================
    # Stage 2: Build a prediction context
    # =========================================================

    historical_experts = sorted(
        {
            expert_id
            for experts in routing_history
            for expert_id in experts
        }
    )

    # Prediction requires at least one candidate other than
    # the current expert. With a real 8-expert Qwen model,
    # multiple observations should produce multiple experts.
    assert len(historical_experts) >= 2

    # Important:
    #
    # Do NOT pass the complete last routing event here.
    #
    # ExpertPredictor intentionally excludes current experts
    # from prediction candidates. Passing the complete routing
    # set can legitimately leave no candidates.
    prediction_context = [
        routing_history[-1][0]
    ]

    # There must be at least one historical expert outside
    # the prediction context.
    assert any(
        expert_id not in prediction_context
        for expert_id in historical_experts
    )

    # =========================================================
    # Stage 3: Generate prediction from learned history
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

    # Current experts must never appear in prediction output.
    assert not (
        set(predicted_experts)
        & set(prediction_context)
    )

    # Every prediction must contain valid score components.
    for prediction in predictions:
        assert prediction.score >= 0.0
        assert prediction.frequency_score >= 0.0
        assert prediction.recency_score >= 0.0
        assert prediction.transition_score >= 0.0

    # =========================================================
    # Stage 4: Materialize predicted experts into cache
    # =========================================================

    for expert_id in predicted_experts:
        cache.insert(
            expert_id,
            location="gpu",
        )

    assert cache.size == len(
        set(predicted_experts)
    )

    for expert_id in predicted_experts:
        assert cache.contains(expert_id)

        entry = cache.lru.peek(expert_id)

        assert entry is not None
        assert entry.expert_id == expert_id
        assert entry.location == "gpu"

        assert entry.hit_count == 0
        assert entry.access_count == 0

    # =========================================================
    # Stage 5: Verify actual cache-hit behavior
    # =========================================================

    prediction_hits, prediction_misses = (
        _lookup_experts(
            cache,
            predicted_experts,
        )
    )

    # Every materialized prediction must hit.
    assert set(prediction_hits) == set(
        predicted_experts
    )

    assert prediction_misses == []

    for expert_id in predicted_experts:
        entry = cache.lru.peek(expert_id)

        assert entry is not None
        assert entry.expert_id == expert_id
        assert entry.location == "gpu"
        assert entry.hit_count == 1
        assert entry.access_count == 1

    # =========================================================
    # Stage 6: Real Qwen routing after prediction
    # =========================================================

    next_events, next_experts = _run_real_qwen(
        model,
        cache,
        input_ids=input_ids,
        step=4,
    )

    assert next_events
    assert next_experts

    assert set(next_experts).issubset(
        valid_experts
    )

    # =========================================================
    # Stage 7: Lookup the real next routing event
    # =========================================================

    real_hits, real_misses = _lookup_experts(
        cache,
        next_experts,
    )

    assert len(real_hits) + len(real_misses) == len(
        next_experts
    )

    # Any real-routing hit must be one of the experts
    # materialized from prediction.
    assert set(real_hits).issubset(
        set(predicted_experts)
    )

    # =========================================================
    # Stage 8: Validate global predictor hit/miss accounting
    # =========================================================

    total_hits = sum(
        stats.hit_count
        for stats in cache.predictor.stats.values()
    )

    total_misses = sum(
        stats.miss_count
        for stats in cache.predictor.stats.values()
    )

    expected_hits = (
        len(prediction_hits)
        + len(real_hits)
    )

    expected_misses = len(real_misses)

    assert total_hits == expected_hits
    assert total_misses == expected_misses

    # The real routing lookup must produce an observable
    # result even when no predicted expert happens to match.
    assert real_hits or real_misses

    # =========================================================
    # Stage 9: Validate hit entries
    # =========================================================

    for expert_id in prediction_hits:
        entry = cache.lru.peek(expert_id)

        assert entry is not None
        assert entry.expert_id == expert_id
        assert entry.location == "gpu"
        assert entry.hit_count >= 1
        assert entry.access_count >= 1

    for expert_id in real_hits:
        entry = cache.lru.peek(expert_id)

        assert entry is not None
        assert entry.expert_id == expert_id
        assert entry.location == "gpu"
        assert entry.hit_count >= 1
        assert entry.access_count >= 1

    # =========================================================
    # Stage 10: Validate real-routing miss accounting
    # =========================================================

    for expert_id in real_misses:
        assert expert_id not in predicted_experts

        stats = cache.predictor.stats.get(
            expert_id
        )

        assert stats is not None
        assert stats.miss_count >= 1
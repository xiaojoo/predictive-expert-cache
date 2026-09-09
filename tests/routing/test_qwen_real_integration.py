import torch

from predictive_cache.cache import PredictiveExpertCache
from predictive_cache.routing import QwenMoeRoutingCapture
from predictive_cache.routing import QwenRoutingBridge
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeForCausalLM,
)


def test_real_qwen_batch_token_alignment():
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

    assert len(events) == 12

    assert cache.predictor.step == 6

    expected_token_ids = [10, 11, 12, 20, 21, 22]

    for layer_id in {0, 1}:
        layer_events = bridge.collector.by_layer(layer_id)

        assert len(layer_events) == 6
        assert [event.token_id for event in layer_events] == (
            expected_token_ids
        )

    for step in range(6):
        step_events = bridge.collector.by_step(step)

        assert len(step_events) == 2
        assert {
            event.token_id
            for event in step_events
        } == {expected_token_ids[step]}

        assert {
            event.layer_id
            for event in step_events
        } == {0, 1}

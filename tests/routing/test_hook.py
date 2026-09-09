from predictive_cache.routing.collector import RoutingEventCollector
from predictive_cache.routing.hook import (
    QwenRoutingHook,
    RoutingHook,
)
from predictive_cache.routing.qwen import QwenRoutingOutput


def test_qwen_routing_hook_collects_output():
    hook = QwenRoutingHook()
    collector = RoutingEventCollector()

    output = QwenRoutingOutput(
        token_id=100,
        layer_expert_ids={
            0: (1, 2, 3),
            1: (4, 5, 6),
        },
    )

    hook(
        step=10,
        output=output,
        collector=collector,
    )

    assert collector.size == 2

    events = collector.events()

    assert events[0].step == 10
    assert events[0].token_id == 100
    assert events[0].layer_id == 0
    assert events[0].expert_ids == (1, 2, 3)

    assert events[1].layer_id == 1
    assert events[1].expert_ids == (4, 5, 6)


def test_qwen_routing_hook_satisfies_protocol():
    hook: RoutingHook = QwenRoutingHook()

    collector = RoutingEventCollector()

    output = QwenRoutingOutput(
        token_id=7,
        layer_expert_ids={
            3: (10, 20),
        },
    )

    hook(
        step=5,
        output=output,
        collector=collector,
    )

    assert collector.size == 1
    assert collector.events()[0].layer_id == 3


def test_qwen_routing_hook_preserves_multiple_steps():
    hook = QwenRoutingHook()
    collector = RoutingEventCollector()

    hook(
        step=1,
        output=QwenRoutingOutput(
            token_id=10,
            layer_expert_ids={
                0: (1, 2),
            },
        ),
        collector=collector,
    )

    hook(
        step=2,
        output=QwenRoutingOutput(
            token_id=11,
            layer_expert_ids={
                0: (3, 4),
            },
        ),
        collector=collector,
    )

    assert collector.by_step(1)[0].expert_ids == (1, 2)
    assert collector.by_step(2)[0].expert_ids == (3, 4)

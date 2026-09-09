import pytest

from predictive_cache.routing.collector import RoutingEventCollector
from predictive_cache.routing.models import RoutingEvent
from predictive_cache.routing.qwen import (
    DefaultQwenRoutingAdapter,
    QwenRoutingAdapter,
    QwenRoutingOutput,
    collect_qwen_routing_output,
    collect_routing_event,
    collect_routing_events,
    create_routing_event,
)


def test_create_routing_event_from_list():
    event = create_routing_event(
        step=10,
        token_id=100,
        layer_id=12,
        expert_ids=[3, 17, 41],
    )

    assert event == RoutingEvent(
        step=10,
        token_id=100,
        layer_id=12,
        expert_ids=(3, 17, 41),
    )


def test_create_routing_event_from_tuple():
    event = create_routing_event(
        step=1,
        token_id=2,
        layer_id=3,
        expert_ids=(4, 8),
    )

    assert event.expert_ids == (4, 8)


def test_collect_routing_event():
    collector = RoutingEventCollector()

    event = collect_routing_event(
        collector,
        step=10,
        token_id=100,
        layer_id=12,
        expert_ids=[3, 17, 41],
    )

    assert event == RoutingEvent(
        step=10,
        token_id=100,
        layer_id=12,
        expert_ids=(3, 17, 41),
    )

    assert collector.events() == (event,)


def test_qwen_adapter_preserves_routing_validation():
    with pytest.raises(ValueError, match="expert_ids must not be empty"):
        create_routing_event(
            step=0,
            token_id=0,
            layer_id=0,
            expert_ids=[],
        )


def test_qwen_adapter_rejects_duplicate_experts():
    with pytest.raises(
        ValueError,
        match="expert_ids must not contain duplicates",
    ):
        create_routing_event(
            step=0,
            token_id=0,
            layer_id=0,
            expert_ids=[1, 1],
        )

def test_collect_routing_events_for_multiple_layers():
    collector = RoutingEventCollector()

    events = collect_routing_events(
        collector,
        step=10,
        token_id=100,
        layer_expert_ids={
            0: [1, 2, 3],
            1: [4, 5, 6],
            2: [7, 8, 9],
        },
    )

    assert events == (
        RoutingEvent(
            step=10,
            token_id=100,
            layer_id=0,
            expert_ids=(1, 2, 3),
        ),
        RoutingEvent(
            step=10,
            token_id=100,
            layer_id=1,
            expert_ids=(4, 5, 6),
        ),
        RoutingEvent(
            step=10,
            token_id=100,
            layer_id=2,
            expert_ids=(7, 8, 9),
        ),
    )

    assert collector.events() == events


def test_collect_routing_events_preserves_layer_order():
    collector = RoutingEventCollector()

    events = collect_routing_events(
        collector,
        step=3,
        token_id=7,
        layer_expert_ids={
            5: [10, 11],
            2: [20, 21],
            8: [30, 31],
        },
    )

    assert [event.layer_id for event in events] == [5, 2, 8]


def test_collect_routing_events_empty_input():
    collector = RoutingEventCollector()

    events = collect_routing_events(
        collector,
        step=3,
        token_id=7,
        layer_expert_ids={},
    )

    assert events == ()
    assert collector.events() == ()

def test_qwen_routing_output_normalizes_expert_ids():
    output = QwenRoutingOutput(
        token_id=100,
        layer_expert_ids={
            0: [1, 2, 3],
            1: [4, 5, 6],
        },
    )

    assert output.layer_expert_ids == {
        0: (1, 2, 3),
        1: (4, 5, 6),
    }


def test_qwen_routing_output_rejects_negative_token():
    with pytest.raises(ValueError, match="token_id must be >= 0"):
        QwenRoutingOutput(
            token_id=-1,
            layer_expert_ids={},
        )


def test_qwen_routing_output_rejects_negative_layer():
    with pytest.raises(ValueError, match="layer_id must be >= 0"):
        QwenRoutingOutput(
            token_id=1,
            layer_expert_ids={
                -1: [1, 2],
            },
        )


def test_collect_qwen_routing_output():
    collector = RoutingEventCollector()

    output = QwenRoutingOutput(
        token_id=100,
        layer_expert_ids={
            0: [1, 2, 3],
            1: [4, 5, 6],
        },
    )

    events = collect_qwen_routing_output(
        collector,
        step=10,
        output=output,
    )

    assert events == (
        RoutingEvent(
            step=10,
            token_id=100,
            layer_id=0,
            expert_ids=(1, 2, 3),
        ),
        RoutingEvent(
            step=10,
            token_id=100,
            layer_id=1,
            expert_ids=(4, 5, 6),
        ),
    )

    assert collector.events() == events

def test_default_qwen_routing_adapter():
    adapter = DefaultQwenRoutingAdapter()

    output = adapter.adapt(
        token_id=100,
        layer_expert_ids={
            0: [1, 2, 3],
            1: [4, 5, 6],
        },
    )

    assert output == QwenRoutingOutput(
        token_id=100,
        layer_expert_ids={
            0: (1, 2, 3),
            1: (4, 5, 6),
        },
    )


def test_default_qwen_routing_adapter_satisfies_protocol():
    adapter: QwenRoutingAdapter = DefaultQwenRoutingAdapter()

    output = adapter.adapt(
        token_id=7,
        layer_expert_ids={
            2: (10, 20),
        },
    )

    assert output.token_id == 7
    assert output.layer_expert_ids == {
        2: (10, 20),
    }


def test_default_qwen_routing_adapter_preserves_validation():
    adapter = DefaultQwenRoutingAdapter()

    with pytest.raises(ValueError, match="layer_id must be >= 0"):
        adapter.adapt(
            token_id=1,
            layer_expert_ids={
                -1: [1, 2],
            },
        )
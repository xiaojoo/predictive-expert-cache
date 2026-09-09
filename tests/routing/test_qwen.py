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

def test_default_qwen_routing_adapter_adapts_router_indices():
    import torch

    adapter = DefaultQwenRoutingAdapter()

    router_indices = torch.tensor(
        [
            [3, 7, 12, 20],
            [1, 5, 9, 11],
        ]
    )

    output = adapter.adapt_router_output(
        token_id=100,
        token_position=1,
        layer_id=5,
        router_indices=router_indices,
    )

    assert output.token_id == 100
    assert output.layer_expert_ids == {
        5: (1, 5, 9, 11),
    }

def test_default_qwen_routing_adapter_separates_token_id_and_position():
    import torch

    adapter = DefaultQwenRoutingAdapter()

    router_indices = torch.tensor(
        [
            [2, 4],
            [6, 8],
        ]
    )

    output = adapter.adapt_router_output(
        token_id=10000,
        token_position=1,
        layer_id=12,
        router_indices=router_indices,
    )

    assert output.token_id == 10000
    assert output.layer_expert_ids == {
        12: (6, 8),
    }

def test_default_qwen_routing_adapter_rejects_invalid_router_shape():
    import torch
    import pytest

    adapter = DefaultQwenRoutingAdapter()

    with pytest.raises(ValueError, match="\\[seq_len, top_k\\]"):
        adapter.adapt_router_output(
            token_id=1,
            token_position=0,
            layer_id=0,
            router_indices=torch.tensor([1, 2, 3]),
        )

def test_default_qwen_routing_adapter_rejects_invalid_token_position():
    import torch
    import pytest

    adapter = DefaultQwenRoutingAdapter()

    with pytest.raises(IndexError, match="token_position out of range"):
        adapter.adapt_router_output(
            token_id=1,
            token_position=2,
            layer_id=0,
            router_indices=torch.tensor(
                [
                    [1, 2],
                    [3, 4],
                ]
            ),
        )

def test_default_qwen_routing_adapter_with_real_qwen3_moe_router():
    import torch

    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeTopKRouter,
    )

    config = Qwen3MoeConfig()

    router = Qwen3MoeTopKRouter(config)

    hidden_states = torch.randn(
        4,
        config.hidden_size,
    )

    router_logits, router_scores, router_indices = router(hidden_states)

    assert router_logits.shape == (
        4,
        config.num_experts,
    )

    assert router_scores.shape == (
        4,
        config.num_experts_per_tok,
    )

    assert router_indices.shape == (
        4,
        config.num_experts_per_tok,
    )

    assert router_indices.dtype == torch.int64

    adapter = DefaultQwenRoutingAdapter()

    token_position = 2
    token_id = 10042
    layer_id = 7

    output = adapter.adapt_router_output(
        token_id=token_id,
        token_position=token_position,
        layer_id=layer_id,
        router_indices=router_indices,
    )

    expected_experts = tuple(
        int(expert_id)
        for expert_id in router_indices[token_position].tolist()
    )

    assert output == QwenRoutingOutput(
        token_id=token_id,
        layer_expert_ids={
            layer_id: expected_experts,
        },
    )

    assert len(output.layer_expert_ids[layer_id]) == (
        config.num_experts_per_tok
    )

    assert all(
        0 <= expert_id < config.num_experts
        for expert_id in output.layer_expert_ids[layer_id]
    )

def test_default_qwen_routing_adapter_real_router_preserves_token_position():
    import torch

    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeTopKRouter,
    )

    config = Qwen3MoeConfig()
    router = Qwen3MoeTopKRouter(config)

    hidden_states = torch.randn(4, config.hidden_size)
    _, _, router_indices = router(hidden_states)

    adapter = DefaultQwenRoutingAdapter()

    for token_position in range(hidden_states.shape[0]):
        output = adapter.adapt_router_output(
            token_id=10000 + token_position,
            token_position=token_position,
            layer_id=7,
            router_indices=router_indices,
        )

        expected = tuple(
            int(expert_id)
            for expert_id in router_indices[token_position].tolist()
        )

        assert output.token_id == 10000 + token_position
        assert output.layer_expert_ids == {
            7: expected,
        }


def test_default_qwen_routing_adapter_real_router_handles_last_token():
    import torch

    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeTopKRouter,
    )

    config = Qwen3MoeConfig()
    router = Qwen3MoeTopKRouter(config)

    hidden_states = torch.randn(4, config.hidden_size)
    _, _, router_indices = router(hidden_states)

    last_position = hidden_states.shape[0] - 1

    output = DefaultQwenRoutingAdapter().adapt_router_output(
        token_id=9999,
        token_position=last_position,
        layer_id=12,
        router_indices=router_indices,
    )

    assert output.token_id == 9999
    assert output.layer_expert_ids == {
        12: tuple(
            int(expert_id)
            for expert_id in router_indices[last_position].tolist()
        )
    }


def test_default_qwen_routing_adapter_real_router_rejects_out_of_range_position():
    import torch
    import pytest

    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeTopKRouter,
    )

    config = Qwen3MoeConfig()
    router = Qwen3MoeTopKRouter(config)

    hidden_states = torch.randn(4, config.hidden_size)
    _, _, router_indices = router(hidden_states)

    with pytest.raises(IndexError, match="token_position out of range"):
        DefaultQwenRoutingAdapter().adapt_router_output(
            token_id=1,
            token_position=hidden_states.shape[0],
            layer_id=0,
            router_indices=router_indices,
        )

def test_default_qwen_routing_adapter_real_router_preserves_expert_order_and_ignores_scores():
    import torch

    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeTopKRouter,
    )

    config = Qwen3MoeConfig()
    router = Qwen3MoeTopKRouter(config)

    hidden_states = torch.randn(4, config.hidden_size)

    router_logits, router_scores, router_indices = router(hidden_states)

    adapter = DefaultQwenRoutingAdapter()

    token_position = 1
    layer_id = 5

    output = adapter.adapt_router_output(
        token_id=1234,
        token_position=token_position,
        layer_id=layer_id,
        router_indices=router_indices,
    )

    expected_experts = tuple(
        int(expert_id)
        for expert_id in router_indices[token_position].tolist()
    )

    expected_scores = router_scores[token_position].tolist()

    assert len(expected_experts) == config.num_experts_per_tok
    assert len(expected_scores) == config.num_experts_per_tok

    assert output.layer_expert_ids[layer_id] == expected_experts

    assert output.layer_expert_ids[layer_id] != tuple(
        int(score)
        for score in expected_scores
    )

    assert all(
        0 <= expert_id < config.num_experts
        for expert_id in output.layer_expert_ids[layer_id]
    )

    assert tuple(
        output.layer_expert_ids[layer_id]
    ) == expected_experts

def test_default_qwen_routing_adapter_rejects_non_2d_router_indices():
    import torch
    import pytest

    adapter = DefaultQwenRoutingAdapter()

    with pytest.raises(
        ValueError,
        match=r"router_indices must have shape \[seq_len, top_k\]",
    ):
        adapter.adapt_router_output(
            token_id=1,
            token_position=0,
            layer_id=0,
            router_indices=torch.tensor([1, 2, 3]),
        )


def test_default_qwen_routing_adapter_preserves_real_router_token_rows():
    import torch

    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeTopKRouter,
    )

    config = Qwen3MoeConfig()
    router = Qwen3MoeTopKRouter(config)

    hidden_states = torch.randn(3, config.hidden_size)
    _, _, router_indices = router(hidden_states)

    adapter = DefaultQwenRoutingAdapter()

    outputs = tuple(
        adapter.adapt_router_output(
            token_id=100 + position,
            token_position=position,
            layer_id=3,
            router_indices=router_indices,
        )
        for position in range(hidden_states.shape[0])
    )

    for position, output in enumerate(outputs):
        assert output.token_id == 100 + position
        assert output.layer_expert_ids[3] == tuple(
            int(expert_id)
            for expert_id in router_indices[position].tolist()
        )

def test_qwen_routing_output_collects_multiple_real_qwen_router_layers():
    import torch

    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeTopKRouter,
    )

    config = Qwen3MoeConfig()

    router_layer_0 = Qwen3MoeTopKRouter(config)
    router_layer_1 = Qwen3MoeTopKRouter(config)

    hidden_states = torch.randn(2, config.hidden_size)

    _, _, indices_0 = router_layer_0(hidden_states)
    _, _, indices_1 = router_layer_1(hidden_states)

    token_position = 1

    experts_0 = tuple(
        int(expert_id)
        for expert_id in indices_0[token_position].tolist()
    )
    experts_1 = tuple(
        int(expert_id)
        for expert_id in indices_1[token_position].tolist()
    )

    output = QwenRoutingOutput(
        token_id=4242,
        layer_expert_ids={
            0: experts_0,
            1: experts_1,
        },
    )

    assert output.token_id == 4242
    assert output.layer_expert_ids == {
        0: experts_0,
        1: experts_1,
    }

    collector = RoutingEventCollector()

    events = collect_qwen_routing_output(
        collector,
        step=7,
        output=output,
    )

    assert len(events) == 2

    assert events[0] == RoutingEvent(
        step=7,
        token_id=4242,
        layer_id=0,
        expert_ids=experts_0,
    )

    assert events[1] == RoutingEvent(
        step=7,
        token_id=4242,
        layer_id=1,
        expert_ids=experts_1,
    )

    assert tuple(collector.events()) == events

def test_qwen_routing_output_preserves_real_router_rows_per_token():
    import torch

    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeTopKRouter,
    )

    config = Qwen3MoeConfig()
    router = Qwen3MoeTopKRouter(config)

    hidden_states = torch.randn(4, config.hidden_size)

    _, _, router_indices = router(hidden_states)

    assert router_indices.ndim == 2
    assert router_indices.shape == (
        hidden_states.shape[0],
        config.num_experts_per_tok,
    )

    token_position_0 = 0
    token_position_3 = 3

    experts_0 = tuple(
        int(expert_id)
        for expert_id in router_indices[token_position_0].tolist()
    )
    experts_3 = tuple(
        int(expert_id)
        for expert_id in router_indices[token_position_3].tolist()
    )

    output_0 = DefaultQwenRoutingAdapter().adapt_router_output(
        token_id=1000,
        token_position=token_position_0,
        layer_id=0,
        router_indices=router_indices,
    )

    output_3 = DefaultQwenRoutingAdapter().adapt_router_output(
        token_id=1003,
        token_position=token_position_3,
        layer_id=0,
        router_indices=router_indices,
    )

    assert output_0.token_id == 1000
    assert output_0.layer_expert_ids == {
        0: experts_0,
    }

    assert output_3.token_id == 1003
    assert output_3.layer_expert_ids == {
        0: experts_3,
    }

    assert output_0.layer_expert_ids[0] == experts_0
    assert output_3.layer_expert_ids[0] == experts_3

def test_qwen_router_expert_ids_are_bounded_and_order_preserved():
    import torch

    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeTopKRouter,
    )

    config = Qwen3MoeConfig()
    router = Qwen3MoeTopKRouter(config)

    hidden_states = torch.randn(3, config.hidden_size)

    _, _, router_indices = router(hidden_states)

    assert router_indices.shape == (
        hidden_states.shape[0],
        config.num_experts_per_tok,
    )

    for token_position in range(hidden_states.shape[0]):
        raw_expert_ids = router_indices[token_position].tolist()

        assert len(raw_expert_ids) == config.num_experts_per_tok
        assert all(
            0 <= int(expert_id) < config.num_experts
            for expert_id in raw_expert_ids
        )

        expected_expert_ids = tuple(int(expert_id) for expert_id in raw_expert_ids)

        output = DefaultQwenRoutingAdapter().adapt_router_output(
            token_id=9000 + token_position,
            token_position=token_position,
            layer_id=2,
            router_indices=router_indices,
        )

        assert output.layer_expert_ids[2] == expected_expert_ids

        collector = RoutingEventCollector()

        events = collect_qwen_routing_output(
            collector,
            step=12,
            output=output,
        )

        assert events == (
            RoutingEvent(
                step=12,
                token_id=9000 + token_position,
                layer_id=2,
                expert_ids=expected_expert_ids,
            ),
        )

def test_qwen_router_token_position_boundary_matches_real_router_rows():
    import torch

    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeTopKRouter,
    )

    config = Qwen3MoeConfig()
    router = Qwen3MoeTopKRouter(config)

    hidden_states = torch.randn(5, config.hidden_size)

    _, _, router_indices = router(hidden_states)

    adapter = DefaultQwenRoutingAdapter()

    first = adapter.adapt_router_output(
        token_id=100,
        token_position=0,
        layer_id=0,
        router_indices=router_indices,
    )

    last = adapter.adapt_router_output(
        token_id=104,
        token_position=hidden_states.shape[0] - 1,
        layer_id=0,
        router_indices=router_indices,
    )

    assert first.layer_expert_ids[0] == tuple(
        int(expert_id)
        for expert_id in router_indices[0].tolist()
    )

    assert last.layer_expert_ids[0] == tuple(
        int(expert_id)
        for expert_id in router_indices[-1].tolist()
    )

    assert first.token_id == 100
    assert last.token_id == 104

    with pytest.raises(IndexError, match="token_position out of range"):
        adapter.adapt_router_output(
            token_id=105,
            token_position=hidden_states.shape[0],
            layer_id=0,
            router_indices=router_indices,
        )

    with pytest.raises(IndexError, match="token_position out of range"):
        adapter.adapt_router_output(
            token_id=106,
            token_position=-1,
            layer_id=0,
            router_indices=router_indices,
        )

def test_qwen_router_token_position_boundary_matches_real_router_rows():
    import torch

    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeTopKRouter,
    )

    config = Qwen3MoeConfig()
    router = Qwen3MoeTopKRouter(config)

    hidden_states = torch.randn(5, config.hidden_size)

    _, _, router_indices = router(hidden_states)

    adapter = DefaultQwenRoutingAdapter()

    first = adapter.adapt_router_output(
        token_id=100,
        token_position=0,
        layer_id=0,
        router_indices=router_indices,
    )

    last = adapter.adapt_router_output(
        token_id=104,
        token_position=hidden_states.shape[0] - 1,
        layer_id=0,
        router_indices=router_indices,
    )

    assert first.layer_expert_ids[0] == tuple(
        int(expert_id)
        for expert_id in router_indices[0].tolist()
    )

    assert last.layer_expert_ids[0] == tuple(
        int(expert_id)
        for expert_id in router_indices[-1].tolist()
    )

    assert first.token_id == 100
    assert last.token_id == 104

    with pytest.raises(IndexError, match="token_position out of range"):
        adapter.adapt_router_output(
            token_id=105,
            token_position=hidden_states.shape[0],
            layer_id=0,
            router_indices=router_indices,
        )

    with pytest.raises(IndexError, match="token_position out of range"):
        adapter.adapt_router_output(
            token_id=106,
            token_position=-1,
            layer_id=0,
            router_indices=router_indices,
        )

def test_qwen_real_router_to_routing_event_end_to_end():
    import torch

    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeTopKRouter,
    )

    config = Qwen3MoeConfig()
    router = Qwen3MoeTopKRouter(config)

    hidden_states = torch.randn(3, config.hidden_size)

    _, _, router_indices = router(hidden_states)

    adapter = DefaultQwenRoutingAdapter()
    collector = RoutingEventCollector()

    token_position = 2
    token_id = 5002
    layer_id = 4
    step = 17

    expected_expert_ids = tuple(
        int(expert_id)
        for expert_id in router_indices[token_position].tolist()
    )

    output = adapter.adapt_router_output(
        token_id=token_id,
        token_position=token_position,
        layer_id=layer_id,
        router_indices=router_indices,
    )

    events = collect_qwen_routing_output(
        collector,
        step=step,
        output=output,
    )

    expected_event = RoutingEvent(
        step=step,
        token_id=token_id,
        layer_id=layer_id,
        expert_ids=expected_expert_ids,
    )

    assert output.token_id == token_id
    assert output.layer_expert_ids == {
        layer_id: expected_expert_ids,
    }

    assert events == (expected_event,)
    assert collector.events() == (expected_event,)

def test_qwen_real_router_preserves_event_order_across_steps():
    import torch

    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeTopKRouter,
    )

    config = Qwen3MoeConfig()
    router = Qwen3MoeTopKRouter(config)

    hidden_states = torch.randn(2, config.hidden_size)

    _, _, router_indices = router(hidden_states)

    adapter = DefaultQwenRoutingAdapter()
    collector = RoutingEventCollector()

    expected_events = []

    for step, token_position in enumerate((0, 1), start=20):
        token_id = 6000 + token_position
        layer_id = token_position

        expert_ids = tuple(
            int(expert_id)
            for expert_id in router_indices[token_position].tolist()
        )

        output = adapter.adapt_router_output(
            token_id=token_id,
            token_position=token_position,
            layer_id=layer_id,
            router_indices=router_indices,
        )

        events = collect_qwen_routing_output(
            collector,
            step=step,
            output=output,
        )

        expected_event = RoutingEvent(
            step=step,
            token_id=token_id,
            layer_id=layer_id,
            expert_ids=expert_ids,
        )

        assert events == (expected_event,)
        expected_events.append(expected_event)

    assert collector.events() == tuple(expected_events)

def test_qwen_real_router_aggregates_multiple_layers_for_same_token():
    import torch

    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeTopKRouter,
    )

    config = Qwen3MoeConfig()

    routers = [
        Qwen3MoeTopKRouter(config),
        Qwen3MoeTopKRouter(config),
        Qwen3MoeTopKRouter(config),
    ]

    hidden_states = torch.randn(2, config.hidden_size)
    token_position = 1
    token_id = 7001
    step = 25

    adapter = DefaultQwenRoutingAdapter()
    collector = RoutingEventCollector()

    layer_expert_ids = {}

    for layer_id, router in enumerate(routers):
        _, _, router_indices = router(hidden_states)

        expert_ids = tuple(
            int(expert_id)
            for expert_id in router_indices[token_position].tolist()
        )

        layer_expert_ids[layer_id] = expert_ids

    output = QwenRoutingOutput(
        token_id=token_id,
        layer_expert_ids=layer_expert_ids,
    )

    events = collect_qwen_routing_output(
        collector,
        step=step,
        output=output,
    )

    assert len(events) == len(routers)

    for layer_id, event in enumerate(events):
        assert event == RoutingEvent(
            step=step,
            token_id=token_id,
            layer_id=layer_id,
            expert_ids=layer_expert_ids[layer_id],
        )

    assert collector.events() == events

def test_qwen_real_router_all_layers_keep_same_token_id_and_step():
    import torch

    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeTopKRouter,
    )

    config = Qwen3MoeConfig()
    hidden_states = torch.randn(2, config.hidden_size)

    collector = RoutingEventCollector()
    output_layers = {}

    for layer_id in range(3):
        router = Qwen3MoeTopKRouter(config)
        _, _, router_indices = router(hidden_states)

        output_layers[layer_id] = tuple(
            int(expert_id)
            for expert_id in router_indices[0].tolist()
        )

    output = QwenRoutingOutput(
        token_id=8000,
        layer_expert_ids=output_layers,
    )

    events = collect_qwen_routing_output(
        collector,
        step=31,
        output=output,
    )

    assert len(events) == 3

    for event in events:
        assert event.step == 31
        assert event.token_id == 8000
        assert event.layer_id in output_layers
        assert event.expert_ids == output_layers[event.layer_id]


def test_qwen_real_router_multiple_tokens_and_layers_do_not_cross_contaminate():
    import torch

    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeTopKRouter,
    )

    config = Qwen3MoeConfig()
    hidden_states = torch.randn(3, config.hidden_size)

    router = Qwen3MoeTopKRouter(config)
    _, _, router_indices = router(hidden_states)

    adapter = DefaultQwenRoutingAdapter()
    collector = RoutingEventCollector()

    expected_events = []

    for token_position in (0, 2):
        token_id = 8100 + token_position

        for layer_id in (0, 2):
            expert_ids = tuple(
                int(expert_id)
                for expert_id in router_indices[token_position].tolist()
            )

            output = adapter.adapt_router_output(
                token_id=token_id,
                token_position=token_position,
                layer_id=layer_id,
                router_indices=router_indices,
            )

            events = collect_qwen_routing_output(
                collector,
                step=40 + token_position,
                output=output,
            )

            expected_event = RoutingEvent(
                step=40 + token_position,
                token_id=token_id,
                layer_id=layer_id,
                expert_ids=expert_ids,
            )

            assert events == (expected_event,)
            expected_events.append(expected_event)

    assert collector.events() == tuple(expected_events)


def test_qwen_real_router_top_k_and_expert_range_are_preserved_end_to_end():
    import torch

    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeTopKRouter,
    )

    config = Qwen3MoeConfig()
    router = Qwen3MoeTopKRouter(config)

    hidden_states = torch.randn(4, config.hidden_size)

    _, _, router_indices = router(hidden_states)

    assert router_indices.shape[1] == config.num_experts_per_tok

    adapter = DefaultQwenRoutingAdapter()
    collector = RoutingEventCollector()

    output = adapter.adapt_router_output(
        token_id=9000,
        token_position=3,
        layer_id=7,
        router_indices=router_indices,
    )

    events = collect_qwen_routing_output(
        collector,
        step=50,
        output=output,
    )

    expert_ids = output.layer_expert_ids[7]

    assert len(expert_ids) == config.num_experts_per_tok
    assert all(
        0 <= expert_id < config.num_experts
        for expert_id in expert_ids
    )

    assert events == (
        RoutingEvent(
            step=50,
            token_id=9000,
            layer_id=7,
            expert_ids=expert_ids,
        ),
    )

def test_qwen_routing_api_is_exported_from_routing_package():
    from predictive_cache.routing import (
        DefaultQwenRoutingAdapter,
        QwenRoutingAdapter,
        QwenRoutingOutput,
        collect_qwen_routing_output,
        collect_routing_event,
        collect_routing_events,
        create_routing_event,
    )

    assert create_routing_event is not None
    assert collect_routing_event is not None
    assert collect_routing_events is not None
    assert QwenRoutingOutput is not None
    assert QwenRoutingAdapter is not None
    assert DefaultQwenRoutingAdapter is not None
    assert collect_qwen_routing_output is not None
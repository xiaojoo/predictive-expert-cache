import torch

from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeTopKRouter,
)

from predictive_cache.cache import PredictiveExpertCache
from predictive_cache.prefetch import (
    PrefetchEngine,
    PrefetchPipeline,
    PrefetchSource,
    PrefetchTarget,
)
from predictive_cache.routing import (
    QwenRoutingBridge,
    QwenRoutingOutput,
    RoutingEventCollector,
)
from predictive_cache.scheduler import ExpertScheduler


def test_qwen_routing_bridge_observes_one_predictor_step_for_multiple_layers():
    cache = PredictiveExpertCache()
    collector = RoutingEventCollector()

    bridge = QwenRoutingBridge(
        cache,
        collector=collector,
    )

    output = QwenRoutingOutput(
        token_id=7,
        layer_expert_ids={
            0: (1, 3, 5),
            1: (2, 3, 7),
        },
    )

    experts = bridge.observe(
        step=0,
        output=output,
    )

    assert experts == (1, 3, 5, 2, 7)

    assert cache.predictor.step == 1

    events = collector.events()

    assert len(events) == 2

    assert events[0].layer_id == 0
    assert events[0].expert_ids == (1, 3, 5)

    assert events[1].layer_id == 1
    assert events[1].expert_ids == (2, 3, 7)


def test_qwen_routing_bridge_preserves_predictor_transition_steps():
    cache = PredictiveExpertCache()

    bridge = QwenRoutingBridge(cache)

    bridge.observe(
        step=0,
        output=QwenRoutingOutput(
            token_id=0,
            layer_expert_ids={
                0: (1, 2),
                1: (3, 4),
            },
        ),
    )

    bridge.observe(
        step=1,
        output=QwenRoutingOutput(
            token_id=1,
            layer_expert_ids={
                0: (2, 5),
                1: (4, 6),
            },
        ),
    )

    assert cache.predictor.step == 2

    assert cache.predictor.stats[1].frequency == 1
    assert cache.predictor.stats[2].frequency == 2
    assert cache.predictor.stats[3].frequency == 1
    assert cache.predictor.stats[4].frequency == 2
    assert cache.predictor.stats[5].frequency == 1
    assert cache.predictor.stats[6].frequency == 1


def test_qwen_real_router_to_cache_bridge():
    config = Qwen3MoeConfig()

    router = Qwen3MoeTopKRouter(config)

    hidden_states = torch.randn(
        2,
        config.hidden_size,
    )

    _, _, router_indices = router(hidden_states)

    cache = PredictiveExpertCache()

    bridge = QwenRoutingBridge(cache)

    output = bridge.adapt_and_observe(
        step=0,
        token_id=42,
        token_position=0,
        layer_id=0,
        router_indices=router_indices,
    )

    assert output.token_id == 42
    assert output.layer_expert_ids[0] == tuple(
        int(value)
        for value in router_indices[0].tolist()
    )

    assert cache.predictor.step == 1

    assert cache.predictor.recent

    assert all(
        0 <= expert_id < config.num_experts
        for expert_id in output.layer_expert_ids[0]
    )


def test_qwen_bridge_can_drive_scheduler():
    config = Qwen3MoeConfig()

    router = Qwen3MoeTopKRouter(config)

    hidden_states = torch.randn(
        2,
        config.hidden_size,
    )

    _, _, router_indices = router(hidden_states)

    cache = PredictiveExpertCache()

    bridge = QwenRoutingBridge(cache)

    bridge.adapt_and_observe(
        step=0,
        token_id=0,
        token_position=0,
        layer_id=0,
        router_indices=router_indices,
    )

    bridge.adapt_and_observe(
        step=1,
        token_id=1,
        token_position=1,
        layer_id=0,
        router_indices=router_indices,
    )

    predictions = cache.prefetch_candidates(
        output_experts := list(
            router_indices[1].tolist()
        )
    )

    assert isinstance(predictions, list)

    assert all(
        prediction.expert_id not in output_experts
        for prediction in predictions
    )

def test_qwen_real_router_multiple_layers_advance_one_predictor_step():
    config = Qwen3MoeConfig()

    router = Qwen3MoeTopKRouter(config)

    hidden_states = torch.randn(
        2,
        config.hidden_size,
    )

    _, _, router_indices = router(hidden_states)

    cache = PredictiveExpertCache()
    collector = RoutingEventCollector()

    bridge = QwenRoutingBridge(
        cache,
        collector=collector,
    )

    output = bridge.adapt_and_observe_step(
        step=0,
        token_id=42,
        token_position=0,
        layer_router_indices={
            0: router_indices,
            1: router_indices,
        },
    )

    assert cache.predictor.step == 1

    assert set(output.layer_expert_ids) == {0, 1}

    assert output.layer_expert_ids[0] == tuple(
        int(value)
        for value in router_indices[0].tolist()
    )

    assert output.layer_expert_ids[1] == tuple(
        int(value)
        for value in router_indices[0].tolist()
    )

    events = collector.events()

    assert len(events) == 2
    assert events[0].layer_id == 0
    assert events[1].layer_id == 1

def test_qwen_bridge_to_prefetch_pipeline_end_to_end():
    cache = PredictiveExpertCache()
    bridge = QwenRoutingBridge(cache)

    # Build predictor history through the same bridge used by
    # model routing integration.
    bridge.observe(
        step=0,
        output=QwenRoutingOutput(
            token_id=0,
            layer_expert_ids={
                0: (1,),
            },
        ),
    )

    bridge.observe(
        step=1,
        output=QwenRoutingOutput(
            token_id=1,
            layer_expert_ids={
                0: (2,),
            },
        ),
    )

    bridge.observe(
        step=2,
        output=QwenRoutingOutput(
            token_id=2,
            layer_expert_ids={
                0: (1,),
            },
        ),
    )

    bridge.observe(
        step=3,
        output=QwenRoutingOutput(
            token_id=3,
            layer_expert_ids={
                0: (3,),
            },
        ),
    )

    loaded = []

    def loader(task):
        loaded.append(task)

    scheduler = ExpertScheduler(cache)

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )

    pipeline = PrefetchPipeline(
        scheduler=scheduler,
        engine=engine,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
    )

    try:
        result = pipeline.process([1])

        assert result.current_experts == [1]

        assert result.scheduled_requests

        scheduled_ids = {
            request.expert_id
            for request in result.scheduled_requests
        }

        assert {2, 3}.issubset(scheduled_ids)

        assert result.submitted_tasks

        task_by_id = {
            task.expert_id: task
            for task in result.submitted_tasks
        }

        assert 2 in task_by_id
        assert 3 in task_by_id

        for task in result.submitted_tasks:
            assert task.source == PrefetchSource.NVME
            assert task.target == PrefetchTarget.RAM
            assert 0.0 <= task.confidence <= 1.0
            assert task.estimated_distance >= 0
            assert task.priority >= 0.0

    finally:
        pipeline.stop()
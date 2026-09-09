import torch

from predictive_cache.cache import PredictiveExpertCache
from predictive_cache.routing import QwenMoeRoutingCapture
from predictive_cache.routing import QwenRoutingBridge
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeForCausalLM,
)
from predictive_cache.prefetch import (
    PrefetchEngine,
    PrefetchPipeline,
    PrefetchSource,
    PrefetchStatus,
    PrefetchTarget,
)
from predictive_cache.prefetch.admission import (
    AdmissionController,
    AdmissionReason,
)
from predictive_cache.scheduler import ExpertScheduler
import time

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

def test_real_qwen_routing_to_prefetch_pipeline():
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

    current_experts = sorted(
        {
            expert_id
            for event in events
            if event.step == 5
            for expert_id in event.expert_ids
        }
    )

    assert current_experts

    scheduler = ExpertScheduler(cache)

    loaded = []

    def loader(task):
        loaded.append(task)

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
        result = pipeline.process(current_experts)

        assert result.current_experts == current_experts
        assert result.scheduled_requests
        assert result.submitted_tasks

        scheduled_ids = {
            request.expert_id
            for request in result.scheduled_requests
        }

        submitted_ids = {
            task.expert_id
            for task in result.submitted_tasks
        }

        assert scheduled_ids <= submitted_ids

        for task in result.submitted_tasks:
            assert task.source == PrefetchSource.NVME
            assert task.target == PrefetchTarget.RAM
            assert 0.0 <= task.confidence <= 1.0
            assert task.estimated_distance >= 0
            assert task.priority >= 0.0

    finally:
        pipeline.stop()

def test_real_qwen_routing_prefetch_tasks_complete():
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

    current_experts = sorted(
        {
            expert_id
            for event in events
            if event.step == 5
            for expert_id in event.expert_ids
        }
    )

    assert current_experts

    scheduler = ExpertScheduler(cache)

    loaded = []

    def loader(task):
        loaded.append(task.expert_id)

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
        result = pipeline.process(current_experts)

        assert result.scheduled_requests
        assert result.submitted_tasks

        submitted_ids = {
            task.expert_id
            for task in result.submitted_tasks
        }

        deadline = time.time() + 2.0

        while time.time() < deadline:
            if submitted_ids.issubset(set(loaded)):
                break
            time.sleep(0.01)

        assert submitted_ids.issubset(set(loaded))

        results = pipeline.results()

        assert len(results) == len(result.submitted_tasks)
        assert all(
            item.status.name == "COMPLETED"
            for item in results
        )
        assert {
            item.expert_id
            for item in results
        } == submitted_ids

    finally:
        pipeline.stop()

def test_real_qwen_routing_admission_to_prefetch_completion():
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

    current_experts = sorted(
        {
            expert_id
            for event in events
            if event.step == 5
            for expert_id in event.expert_ids
        }
    )

    assert current_experts

    scheduler = ExpertScheduler(cache)

    loaded = []

    def loader(task):
        loaded.append(task.expert_id)

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
        result = pipeline.process(current_experts)

        assert result.scheduled_requests
        assert result.admission_decisions
        assert result.submitted_tasks

        accepted_decisions = [
            decision
            for decision in result.admission_decisions
            if decision.admitted
        ]

        assert len(accepted_decisions) == len(
            result.submitted_tasks
        )

        assert result.admission_stats.accepted == len(
            accepted_decisions
        )

        submitted_ids = {
            task.expert_id
            for task in result.submitted_tasks
        }

        deadline = time.time() + 2.0

        while time.time() < deadline:
            results = pipeline.results()

            if len(results) == len(result.submitted_tasks):
                break

            time.sleep(0.01)

        results = pipeline.results()

        assert len(results) == len(result.submitted_tasks)
        assert all(
            item.status == PrefetchStatus.COMPLETED
            for item in results
        )

        assert {
            item.expert_id
            for item in results
        } == submitted_ids

        assert set(loaded) == submitted_ids

    finally:
        pipeline.stop()

def test_real_qwen_routing_admission_reject():
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

    current_experts = sorted(
        {
            expert_id
            for event in events
            if event.step == 5
            for expert_id in event.expert_ids
        }
    )

    assert current_experts

    scheduler = ExpertScheduler(cache)

    # 先确认真实 Qwen 路由确实产生了 prefetch requests。
    requests = scheduler.plan_prefetch_predictions(
        current_experts
    )

    assert requests

    # 动态设置一个必然高于所有真实 request priority 的阈值，
    # 从而稳定触发 Admission REJECT，而不依赖硬编码分数。
    max_priority = max(
        request.priority
        for request in requests
    )

    admission = AdmissionController(
        min_priority=max_priority + 1e-6,
    )

    loaded = []

    def loader(task):
        loaded.append(task.expert_id)

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )

    pipeline = PrefetchPipeline(
        scheduler=scheduler,
        engine=engine,
        admission=admission,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
    )

    try:
        result = pipeline.process(current_experts)

        assert result.scheduled_requests
        assert result.admission_decisions

        # Admission 应该阻断全部 requests。
        assert result.submitted_tasks == []

        assert all(
            not decision.admitted
            for decision in result.admission_decisions
        )

        assert all(
            decision.reason
            == AdmissionReason.REJECT_LOW_PRIORITY
            for decision in result.admission_decisions
        )

        assert (
            result.admission_stats.rejected
            == len(result.admission_decisions)
        )

        assert result.admission_stats.accepted == 0

        # 被 Admission 拒绝的 task 不应该进入 worker。
        assert loaded == []

        assert pipeline.results() == []

    finally:
        pipeline.stop()
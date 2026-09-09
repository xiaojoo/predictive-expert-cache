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

    scheduled_requests = scheduler.plan_prefetch_predictions(current_experts)
    scheduled_ids = {
        request.expert_id
        for request in scheduled_requests
    }

    assert scheduled_ids

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

def test_real_qwen_routing_duplicate_admission():
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
        # ---------------------------------------------------------
        # First submission
        # ---------------------------------------------------------

        first_result = pipeline.process(current_experts)

        assert first_result.scheduled_requests
        assert first_result.admission_decisions
        assert first_result.submitted_tasks

        first_submitted_ids = {
            task.expert_id
            for task in first_result.submitted_tasks
        }

        assert first_submitted_ids

        accepted_first = [
            decision
            for decision in first_result.admission_decisions
            if decision.admitted
        ]

        assert len(accepted_first) == len(
            first_result.submitted_tasks
        )

        # ---------------------------------------------------------
        # Immediately submit the same predictions again.
        #
        # The first batch may still be queued or running. Admission
        # must prevent the same expert from being submitted again.
        # ---------------------------------------------------------

        second_result = pipeline.process(current_experts)

        assert second_result.scheduled_requests
        assert second_result.admission_decisions

        second_submitted_ids = {
            task.expert_id
            for task in second_result.submitted_tasks
        }

        assert not (
            first_submitted_ids
            & second_submitted_ids
        )

        duplicate_decisions = [
            decision
            for decision in second_result.admission_decisions
            if decision.reason
            in {
                AdmissionReason.REJECT_DUPLICATE_QUEUED,
                AdmissionReason.REJECT_DUPLICATE_RUNNING,
            }
        ]

        assert duplicate_decisions

        assert all(
            not decision.admitted
            for decision in duplicate_decisions
        )

        # ---------------------------------------------------------
        # Wait for the first batch to complete.
        # ---------------------------------------------------------

        deadline = time.time() + 2.0

        while time.time() < deadline:
            results = pipeline.results()

            if len(results) == len(
                first_result.submitted_tasks
            ):
                break

            time.sleep(0.01)

        results = pipeline.results()

        assert len(results) == len(
            first_result.submitted_tasks
        )

        assert all(
            item.status == PrefetchStatus.COMPLETED
            for item in results
        )

        assert {
            item.expert_id
            for item in results
        } == first_submitted_ids

        # The loader must have executed each first submission once.
        assert set(loaded) == first_submitted_ids

        # ---------------------------------------------------------
        # Admission statistics must include the duplicate rejects.
        # ---------------------------------------------------------

        assert (
            second_result.admission_stats.rejected
            >= len(duplicate_decisions)
        )

    finally:
        pipeline.stop()

def test_real_qwen_routing_duplicate_admission_after_completion():
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
        # ---------------------------------------------------------
        # First submission
        # ---------------------------------------------------------

        first_result = pipeline.process(current_experts)

        assert first_result.scheduled_requests
        assert first_result.admission_decisions
        assert first_result.submitted_tasks

        first_submitted_ids = {
            task.expert_id
            for task in first_result.submitted_tasks
        }

        assert first_submitted_ids

        # ---------------------------------------------------------
        # Immediate duplicate submission must be rejected.
        # ---------------------------------------------------------

        second_result = pipeline.process(current_experts)

        assert second_result.scheduled_requests
        assert second_result.admission_decisions

        second_duplicate_decisions = [
            decision
            for decision in second_result.admission_decisions
            if decision.reason
            in {
                AdmissionReason.REJECT_DUPLICATE_QUEUED,
                AdmissionReason.REJECT_DUPLICATE_RUNNING,
            }
        ]

        assert second_duplicate_decisions

        assert all(
            not decision.admitted
            for decision in second_duplicate_decisions
        )

        assert second_result.submitted_tasks == []

        # ---------------------------------------------------------
        # Wait until the first batch has completely finished.
        # ---------------------------------------------------------

        deadline = time.time() + 2.0

        while time.time() < deadline:
            results = pipeline.results()

            if len(results) == len(
                first_result.submitted_tasks
            ):
                break

            time.sleep(0.01)

        results = pipeline.results()

        assert len(results) == len(
            first_result.submitted_tasks
        )

        assert all(
            item.status == PrefetchStatus.COMPLETED
            for item in results
        )

        assert {
            item.expert_id
            for item in results
        } == first_submitted_ids

        assert set(loaded) == first_submitted_ids

        # ---------------------------------------------------------
        # Clear completed results before the third submission.
        # This isolates the new submission from the first batch.
        # ---------------------------------------------------------

        pipeline.clear_results()

        assert pipeline.results() == []

        # ---------------------------------------------------------
        # After completion, the same real Qwen predictions must
        # become admissible again.
        # ---------------------------------------------------------

        third_result = pipeline.process(current_experts)

        assert third_result.scheduled_requests
        assert third_result.admission_decisions
        assert third_result.submitted_tasks

        third_submitted_ids = {
            task.expert_id
            for task in third_result.submitted_tasks
        }

        assert third_submitted_ids == first_submitted_ids

        assert all(
            decision.admitted
            for decision in third_result.admission_decisions
            if decision.reason == AdmissionReason.ACCEPT
        )

        assert len(third_result.submitted_tasks) == len(
            [
                decision
                for decision in third_result.admission_decisions
                if decision.admitted
            ]
        )

        # ---------------------------------------------------------
        # Wait for the third batch to complete.
        # ---------------------------------------------------------

        deadline = time.time() + 2.0

        while time.time() < deadline:
            results = pipeline.results()

            if len(results) == len(
                third_result.submitted_tasks
            ):
                break

            time.sleep(0.01)

        results = pipeline.results()

        assert len(results) == len(
            third_result.submitted_tasks
        )

        assert all(
            item.status == PrefetchStatus.COMPLETED
            for item in results
        )

        assert {
            item.expert_id
            for item in results
        } == third_submitted_ids

        # Each expert was loaded once in the first batch and once
        # again after the previous batch completed.
        assert len(loaded) == (
            len(first_submitted_ids)
            + len(third_submitted_ids)
        )

        assert set(loaded) == (
            first_submitted_ids
            | third_submitted_ids
        )

    finally:
        pipeline.stop()

def test_real_qwen_routing_admission_engine_state_consistency():
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
        # ---------------------------------------------------------
        # Initial state
        # ---------------------------------------------------------

        assert pipeline.running
        assert engine.queued_tasks == []
        assert engine.running_tasks == []
        assert pipeline.results() == []

        # ---------------------------------------------------------
        # First admission
        # ---------------------------------------------------------

        first_result = pipeline.process(current_experts)

        assert first_result.scheduled_requests
        assert first_result.admission_decisions
        assert first_result.submitted_tasks

        first_ids = {
            task.expert_id
            for task in first_result.submitted_tasks
        }

        assert first_ids

        assert {
            decision.reason
            for decision in first_result.admission_decisions
            if decision.admitted
        } == {AdmissionReason.ACCEPT}

        assert first_result.admission_stats.accepted == len(
            [
                decision
                for decision in first_result.admission_decisions
                if decision.admitted
            ]
        )

        assert first_result.admission_stats.rejected == len(
            [
                decision
                for decision in first_result.admission_decisions
                if not decision.admitted
            ]
        )

        # At least one task must be visible to the engine
        # immediately as queued or running.
        assert (
            engine.queued_tasks
            or engine.running_tasks
        )

        assert set(engine.queued_tasks) <= first_ids
        assert set(engine.running_tasks) <= first_ids

        # ---------------------------------------------------------
        # Immediate duplicate
        # ---------------------------------------------------------

        duplicate_result = pipeline.process(current_experts)

        duplicate_decisions = [
            decision
            for decision in duplicate_result.admission_decisions
            if decision.reason
            in {
                AdmissionReason.REJECT_DUPLICATE_QUEUED,
                AdmissionReason.REJECT_DUPLICATE_RUNNING,
            }
        ]

        assert duplicate_decisions

        assert all(
            not decision.admitted
            for decision in duplicate_decisions
        )

        assert duplicate_result.submitted_tasks == []

        # The first batch remains the only active batch.
        assert set(engine.queued_tasks).issubset(first_ids)
        assert set(engine.running_tasks).issubset(first_ids)

        # ---------------------------------------------------------
        # Wait for complete execution
        # ---------------------------------------------------------

        deadline = time.time() + 2.0

        while time.time() < deadline:
            results = pipeline.results()

            if len(results) == len(
                first_result.submitted_tasks
            ):
                break

            time.sleep(0.01)

        results = pipeline.results()

        assert len(results) == len(
            first_result.submitted_tasks
        )

        assert all(
            item.status == PrefetchStatus.COMPLETED
            for item in results
        )

        assert {
            item.expert_id
            for item in results
        } == first_ids

        assert set(loaded) == first_ids

        # ---------------------------------------------------------
        # Critical lifecycle invariant:
        #
        # completed tasks must no longer be considered queued
        # or running by the admission engine.
        # ---------------------------------------------------------

        assert engine.queued_tasks == []
        assert engine.running_tasks == []

        # Results are retained independently from active state.
        assert len(pipeline.results()) == len(first_ids)

        # ---------------------------------------------------------
        # Clear historical results.
        #
        # This must not be required for admission eligibility;
        # it only removes completed result history.
        # ---------------------------------------------------------

        pipeline.clear_results()

        assert pipeline.results() == []

        assert engine.queued_tasks == []
        assert engine.running_tasks == []

        # ---------------------------------------------------------
        # Same prediction after completion must be admissible again.
        # ---------------------------------------------------------

        second_result = pipeline.process(current_experts)

        assert second_result.scheduled_requests
        assert second_result.admission_decisions
        assert second_result.submitted_tasks

        second_ids = {
            task.expert_id
            for task in second_result.submitted_tasks
        }

        assert second_ids == first_ids

        assert all(
            decision.admitted
            for decision in second_result.admission_decisions
            if decision.reason == AdmissionReason.ACCEPT
        )

        assert set(engine.queued_tasks).issubset(second_ids)
        assert set(engine.running_tasks).issubset(second_ids)

        # ---------------------------------------------------------
        # Wait for second execution
        # ---------------------------------------------------------

        deadline = time.time() + 2.0

        while time.time() < deadline:
            results = pipeline.results()

            if len(results) == len(
                second_result.submitted_tasks
            ):
                break

            time.sleep(0.01)

        results = pipeline.results()

        assert len(results) == len(
            second_result.submitted_tasks
        )

        assert all(
            item.status == PrefetchStatus.COMPLETED
            for item in results
        )

        assert {
            item.expert_id
            for item in results
        } == second_ids

        # ---------------------------------------------------------
        # Final lifecycle invariant
        # ---------------------------------------------------------

        assert engine.queued_tasks == []
        assert engine.running_tasks == []

        assert len(loaded) == (
            len(first_ids)
            + len(second_ids)
        )

        assert loaded.count(next(iter(first_ids))) == 2

    finally:
        pipeline.stop()

        assert not pipeline.running

def test_real_qwen_routing_prefetch_storage_transfer():
    from predictive_cache.prefetch.storage import (
        create_storage_transfer_executor,
    )
    from predictive_cache.storage.expert_record import ExpertRecord
    from predictive_cache.storage.expert_store import ExpertLocation
    from predictive_cache.storage.in_memory import InMemoryExpertStore

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

    # ---------------------------------------------------------
    # Determine the experts actually predicted by the scheduler.
    # ---------------------------------------------------------

    scheduled_requests = scheduler.plan_prefetch_predictions(
        current_experts
    )

    assert scheduled_requests

    scheduled_ids = {
        request.expert_id
        for request in scheduled_requests
    }

    assert scheduled_ids

    # ---------------------------------------------------------
    # Build source NVME store for predicted experts.
    # ---------------------------------------------------------

    source_store = InMemoryExpertStore()
    target_store = InMemoryExpertStore()

    payloads = {
        expert_id: f"expert-payload-{expert_id}"
        for expert_id in scheduled_ids
    }

    for expert_id, payload in payloads.items():
        source_store.put(
            ExpertRecord(
                expert_id=expert_id,
                location=ExpertLocation.NVME,
                payload=payload,
            )
        )

    for expert_id in scheduled_ids:
        record = source_store.get(expert_id)

        assert record is not None
        assert record.expert_id == expert_id
        assert record.location == ExpertLocation.NVME
        assert record.payload == payloads[expert_id]

    # ---------------------------------------------------------
    # Real storage-backed transfer:
    #
    # NVME ExpertStore -> PrefetchStorage -> TransferExecutor
    # -> RAM ExpertStore
    # ---------------------------------------------------------

    def fallback_loader(task):
        raise AssertionError(
            "storage-backed NVME->RAM transfer should handle "
            "the task without fallback loader"
        )

    transfer_executor = create_storage_transfer_executor(
        source_store,
        target_store,
        fallback_loader,
    )

    engine = PrefetchEngine(
        fallback_loader,
        num_workers=1,
        transfer_executor=transfer_executor,
    )

    pipeline = PrefetchPipeline(
        scheduler=scheduler,
        engine=engine,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
    )

    try:
        # ---------------------------------------------------------
        # Admission + submission
        # ---------------------------------------------------------

        result = pipeline.process(current_experts)

        assert result.current_experts == current_experts
        assert result.scheduled_requests
        assert result.admission_decisions
        assert result.submitted_tasks

        submitted_ids = {
            task.expert_id
            for task in result.submitted_tasks
        }

        assert submitted_ids
        assert submitted_ids <= scheduled_ids

        assert all(
            decision.admitted
            for decision in result.admission_decisions
        )

        # ---------------------------------------------------------
        # Wait for actual storage transfer completion.
        # ---------------------------------------------------------

        deadline = time.time() + 2.0

        while time.time() < deadline:
            results = pipeline.results()

            if len(results) == len(
                result.submitted_tasks
            ):
                break

            time.sleep(0.01)

        results = pipeline.results()

        assert len(results) == len(
            result.submitted_tasks
        )

        assert all(
            item.status == PrefetchStatus.COMPLETED
            for item in results
        )

        assert {
            item.expert_id
            for item in results
        } == submitted_ids

        # ---------------------------------------------------------
        # Critical 6D-1 invariant:
        #
        # Actual payload must travel from NVME source
        # store into RAM target store.
        # ---------------------------------------------------------

        for expert_id in submitted_ids:
            source_record = source_store.get(expert_id)
            target_record = target_store.get(expert_id)

            assert source_record is not None
            assert target_record is not None

            assert source_record.location == ExpertLocation.NVME

            assert target_record.expert_id == expert_id
            assert target_record.location == ExpertLocation.RAM
            assert target_record.payload == payloads[expert_id]

        assert target_store is not source_store

        assert all(
            target_store.contains(expert_id)
            for expert_id in submitted_ids
        )

    finally:
        pipeline.stop()

        assert not pipeline.running
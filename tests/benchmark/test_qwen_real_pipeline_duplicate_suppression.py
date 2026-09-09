from __future__ import annotations

import threading
import time

import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from predictive_cache.cache import PredictiveExpertCache
from predictive_cache.prefetch.engine import PrefetchEngine
from predictive_cache.prefetch.pipeline import PrefetchPipeline
from predictive_cache.prefetch.types import (
    PrefetchSource,
    PrefetchTarget,
    PrefetchTask,
)
from predictive_cache.routing import QwenMoeRoutingCapture
from predictive_cache.routing import QwenRoutingBridge
from predictive_cache.scheduler import ExpertScheduler
from predictive_cache.types import CacheConfig


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


def _wait_for_tasks(
    engine: PrefetchEngine,
    expert_ids: list[int],
    *,
    timeout: float = 1.0,
) -> None:
    deadline = time.perf_counter() + timeout

    for expert_id in expert_ids:
        while time.perf_counter() < deadline:
            if engine.task_status(expert_id) == "completed":
                break
            time.sleep(0.00005)

        assert engine.task_status(expert_id) == "completed"


def _make_inputs() -> list[torch.Tensor]:
    return [
        torch.tensor([[10, 11, 12]], dtype=torch.long),
        torch.tensor([[20, 21, 22]], dtype=torch.long),
        torch.tensor([[30, 31, 32]], dtype=torch.long),
        torch.tensor([[40, 41, 42]], dtype=torch.long),
        torch.tensor([[50, 51, 52]], dtype=torch.long),
        torch.tensor([[60, 61, 62]], dtype=torch.long),
        torch.tensor([[70, 71, 72]], dtype=torch.long),
        torch.tensor([[80, 81, 82]], dtype=torch.long),
    ]


def test_real_qwen_pipeline_duplicate_running_suppression() -> None:
    """
    7-J-1:

        real Qwen routing
            -> Predictor
            -> Scheduler
            -> Pipeline
            -> AdmissionController
            -> Engine

    The same prediction batch is submitted twice while the first
    submission is deliberately held in RUNNING state.

    Required invariants:

    1. Real Qwen routing produces a non-empty current expert set.
    2. The first pipeline invocation admits at least one task.
    3. The loader holds the task in RUNNING state.
    4. A second identical pipeline invocation does not submit
       duplicate-running tasks.
    5. Duplicate-running decisions are explicitly observed.
    6. The original task eventually completes normally.
    """

    model = _build_real_qwen()

    cache = PredictiveExpertCache(
        CacheConfig(
            capacity=8,
            prediction_top_k=3,
        )
    )

    bridge = QwenRoutingBridge(cache)
    capture = QwenMoeRoutingCapture(bridge)

    scheduler = ExpertScheduler(cache)

    loader_started = threading.Event()
    loader_release = threading.Event()

    executions = 0

    def loader(task) -> None:
        nonlocal executions

        executions += 1
        loader_started.set()

        # Hold the first task in RUNNING state so that the next
        # identical pipeline invocation sees duplicate-running work.
        assert loader_release.wait(timeout=5.0)

        cache.insert(
            task.expert_id,
            size_bytes=1024,
            location="memory",
        )

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )

    pipeline = PrefetchPipeline(
        scheduler,
        engine,
        auto_start=True,
    )

    inputs = _make_inputs()

    try:
        # ---------------------------------------------------------
        # Establish real routing history.
        #
        # Real Qwen routing is input-dependent, so the final routing
        # state is not guaranteed to have a schedulable prediction.
        # Select the first real routing state that does.
        # ---------------------------------------------------------
        routed_currents: list[list[int]] = []

        for step in range(4):
            current = _capture_routing(
                model,
                capture,
                inputs[step % len(inputs)],
                step=step,
            )

            cache.observe(current)
            routed_currents.append(current)

        assert routed_currents

        selected_current: list[int] | None = None

        for candidate_current in routed_currents:
            if scheduler.plan_prefetch_predictions(candidate_current):
                selected_current = candidate_current
                break

        assert selected_current is not None

        # ---------------------------------------------------------
        # First submission.
        # ---------------------------------------------------------
        first = pipeline.process(selected_current)

        assert first.scheduled_requests
        assert first.submitted_tasks

        first_ids = {
            task.expert_id
            for task in first.submitted_tasks
        }

        assert first_ids

        # Wait until at least one real loader execution is RUNNING.
        assert loader_started.wait(timeout=5.0)

        running_ids = {
            expert_id
            for expert_id in first_ids
            if engine.task_status(expert_id) == "running"
        }

        assert running_ids

        # ---------------------------------------------------------
        # Identical second submission while the first is RUNNING.
        # ---------------------------------------------------------
        second = pipeline.process(selected_current)

        duplicate_running = [
            decision
            for decision in second.admission_decisions
            if decision.reason.value == "reject_duplicate_running"
        ]

        second_ids = {
            task.expert_id
            for task in second.submitted_tasks
        }

        print()
        print("7-J-1 Real Qwen -> Duplicate Running Suppression")
        print("------------------------------------------------")
        print(f"current experts:        {len(selected_current)}")
        print(f"first scheduled:        {len(first.scheduled_requests)}")
        print(f"first submitted:        {len(first.submitted_tasks)}")
        print(f"running first IDs:      {len(running_ids)}")
        print(f"second scheduled:       {len(second.scheduled_requests)}")
        print(f"second submitted:       {len(second.submitted_tasks)}")
        print(f"duplicate-running:      {len(duplicate_running)}")
        print(f"second submitted IDs:   {len(second_ids)}")

        # The second invocation must not submit an expert already
        # represented by a running task.
        assert not (running_ids & second_ids)

        # At least one duplicate-running admission must be visible.
        assert duplicate_running

        # Release the original loader and verify normal completion.
        loader_release.set()

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if all(
                engine.task_status(expert_id) == "completed"
                for expert_id in first_ids
            ):
                break
            time.sleep(0.01)

        for expert_id in first_ids:
            assert engine.task_status(expert_id) == "completed"

        assert executions == len(first.submitted_tasks)

    finally:
        loader_release.set()
        if engine.running:
            engine.stop(wait=True)


def test_real_qwen_pipeline_duplicate_queued_suppression() -> None:
    """
    7-J-2:

        real Qwen routing
            -> Predictor
            -> Scheduler
            -> Pipeline
            -> AdmissionController
            -> Engine

    A blocker occupies the only Engine worker in RUNNING state.
    The first real-Qwen prefetch therefore remains QUEUED.

    A second identical pipeline invocation must reject that
    prefetch explicitly as duplicate-queued.
    """

    model = _build_real_qwen()

    cache = PredictiveExpertCache(
        CacheConfig(
            capacity=8,
            prediction_top_k=3,
        )
    )

    bridge = QwenRoutingBridge(cache)
    capture = QwenMoeRoutingCapture(bridge)
    scheduler = ExpertScheduler(cache)

    blocker_started = threading.Event()
    blocker_release = threading.Event()

    executed_ids: list[int] = []

    blocker_id = 10_000

    def loader(task) -> None:
        executed_ids.append(task.expert_id)

        if task.expert_id == blocker_id:
            blocker_started.set()
            assert blocker_release.wait(timeout=5.0)

        cache.insert(
            task.expert_id,
            size_bytes=1024,
            location="memory",
        )

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )

    pipeline = PrefetchPipeline(
        scheduler,
        engine,
        auto_start=True,
    )

    inputs = _make_inputs()

    try:
        # ---------------------------------------------------------
        # Establish real routing history.
        #
        # Real Qwen routing is input-dependent, so the final routing
        # state is not guaranteed to have a schedulable prediction.
        # Select the first real routing state that does.
        # ---------------------------------------------------------
        routed_currents: list[list[int]] = []

        for step in range(4):
            current = _capture_routing(
                model,
                capture,
                inputs[step % len(inputs)],
                step=step,
            )

            cache.observe(current)
            routed_currents.append(current)

        assert routed_currents

        selected_current: list[int] | None = None

        for candidate_current in routed_currents:
            if scheduler.plan_prefetch_predictions(candidate_current):
                selected_current = candidate_current
                break

        assert selected_current is not None

        # ---------------------------------------------------------
        # Occupy the sole worker so the real prefetch stays QUEUED.
        # ---------------------------------------------------------
        blocker = PrefetchTask(
            expert_id=blocker_id,
            source=PrefetchSource.NVME,
            target=PrefetchTarget.RAM,
        )

        assert engine.submit(blocker)
        assert blocker_started.wait(timeout=5.0)
        assert engine.task_status(blocker_id) == "running"

        # ---------------------------------------------------------
        # First real-Qwen pipeline invocation.
        # ---------------------------------------------------------
        first = pipeline.process(selected_current)

        assert first.scheduled_requests
        assert first.submitted_tasks

        first_ids = {
            task.expert_id
            for task in first.submitted_tasks
        }

        assert first_ids

        queued_ids = {
            expert_id
            for expert_id in first_ids
            if engine.task_status(expert_id) == "queued"
        }

        assert queued_ids

        # ---------------------------------------------------------
        # Identical second submission while the first is QUEUED.
        # ---------------------------------------------------------
        second = pipeline.process(selected_current)

        duplicate_queued = [
            decision
            for decision in second.admission_decisions
            if decision.reason.value == "reject_duplicate_queued"
        ]

        second_ids = {
            task.expert_id
            for task in second.submitted_tasks
        }

        print()
        print("7-J-2 Real Qwen -> Duplicate Queued Suppression")
        print("------------------------------------------------")
        print(f"current experts:        {len(selected_current)}")
        print(f"first scheduled:        {len(first.scheduled_requests)}")
        print(f"first submitted:        {len(first.submitted_tasks)}")
        print(f"queued first IDs:       {len(queued_ids)}")
        print(f"second scheduled:       {len(second.scheduled_requests)}")
        print(f"second submitted:       {len(second.submitted_tasks)}")
        print(f"duplicate-queued:       {len(duplicate_queued)}")
        print(f"second submitted IDs:   {len(second_ids)}")

        # The queued tasks must not be submitted a second time.
        assert not (queued_ids & second_ids)

        # At least one duplicate-queued admission must be visible.
        assert duplicate_queued

        # No second invocation task may be submitted.
        assert not second_ids

        # The original prefetch must remain queued.
        for expert_id in queued_ids:
            assert engine.task_status(expert_id) == "queued"

        # ---------------------------------------------------------
        # Release the blocker and verify normal completion.
        # ---------------------------------------------------------
        blocker_release.set()

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if all(
                engine.task_status(expert_id) == "completed"
                for expert_id in first_ids
            ):
                break
            time.sleep(0.01)

        assert engine.task_status(blocker_id) == "completed"

        for expert_id in first_ids:
            assert engine.task_status(expert_id) == "completed"

        assert set(executed_ids) >= first_ids

    finally:
        blocker_release.set()
        if engine.running:
            engine.stop(wait=True)


def test_real_qwen_pipeline_queue_full_suppression() -> None:
    """
    7-J-3:

        real Qwen routing
            -> Predictor
            -> Scheduler
            -> Pipeline
            -> AdmissionController
            -> Engine

    The Engine has one worker and a queue capacity of one.

    A blocker occupies the worker, so the first real-Qwen prefetch
    remains QUEUED and fills the queue.

    A second real-Qwen routing result must produce a different
    prefetch candidate. That candidate must then be rejected as
    queue-full rather than being submitted.
    """

    model = _build_real_qwen()

    cache = PredictiveExpertCache(
        CacheConfig(
            capacity=8,
            prediction_top_k=3,
        )
    )

    bridge = QwenRoutingBridge(cache)
    capture = QwenMoeRoutingCapture(bridge)
    scheduler = ExpertScheduler(cache)

    blocker_started = threading.Event()
    blocker_release = threading.Event()

    executed_ids: list[int] = []
    blocker_id = 10_001

    def loader(task) -> None:
        executed_ids.append(task.expert_id)

        if task.expert_id == blocker_id:
            blocker_started.set()
            assert blocker_release.wait(timeout=5.0)

        cache.insert(
            task.expert_id,
            size_bytes=1024,
            location="memory",
        )

    engine = PrefetchEngine(
        loader,
        num_workers=1,
        max_queue_size=1,
    )

    pipeline = PrefetchPipeline(
        scheduler,
        engine,
        auto_start=True,
    )

    inputs = _make_inputs()

    try:
        # ---------------------------------------------------------
        # Build real Qwen routing history.
        # ---------------------------------------------------------
        routed_currents: list[list[int]] = []

        for step, input_ids in enumerate(inputs):
            current = _capture_routing(
                model,
                capture,
                input_ids,
                step=step,
            )
            cache.observe(current)
            routed_currents.append(current)

        assert routed_currents

        # ---------------------------------------------------------
        # Find two real routing states whose scheduler candidates
        # are different. Planning is side-effect free for Engine
        # admission, so this lets us select a valid pair before
        # filling the queue.
        # ---------------------------------------------------------
        selected_first: list[int] | None = None
        selected_second: list[int] | None = None
        first_candidate: int | None = None
        second_candidate: int | None = None

        planned = []

        for current in routed_currents:
            requests = scheduler.plan_prefetch_predictions(current)
            planned.append((current, requests))

        for first_index, (current_a, requests_a) in enumerate(planned):
            if not requests_a:
                continue

            candidate_a = requests_a[0].expert_id

            for second_index, (current_b, requests_b) in enumerate(planned):
                if first_index == second_index or not requests_b:
                    continue

                candidate_b = requests_b[0].expert_id

                if candidate_a != candidate_b:
                    selected_first = current_a
                    selected_second = current_b
                    first_candidate = candidate_a
                    second_candidate = candidate_b
                    break

            if selected_first is not None:
                break

        assert selected_first is not None
        assert selected_second is not None
        assert first_candidate is not None
        assert second_candidate is not None
        assert first_candidate != second_candidate

        # ---------------------------------------------------------
        # Occupy the only worker. The first real prefetch will then
        # remain QUEUED and fill the queue of size one.
        # ---------------------------------------------------------
        blocker = PrefetchTask(
            expert_id=blocker_id,
            source=PrefetchSource.NVME,
            target=PrefetchTarget.RAM,
        )

        assert engine.submit(blocker)
        assert blocker_started.wait(timeout=5.0)
        assert engine.task_status(blocker_id) == "running"

        # ---------------------------------------------------------
        # First real-Qwen pipeline invocation.
        # ---------------------------------------------------------
        first = pipeline.process(selected_first)

        assert first.scheduled_requests
        assert first.submitted_tasks

        first_ids = {
            task.expert_id
            for task in first.submitted_tasks
        }

        assert first_ids
        assert first_candidate in first_ids
        assert all(
            engine.task_status(expert_id) == "queued"
            for expert_id in first_ids
        )
        assert engine.queue_full

        # ---------------------------------------------------------
        # Second real-Qwen pipeline invocation with a different
        # routing state / candidate.
        #
        # Because the queue is full and the candidate is different,
        # this must hit REJECT_QUEUE_FULL rather than duplicate-
        # queued.
        # ---------------------------------------------------------
        second = pipeline.process(selected_second)

        queue_full_decisions = [
            decision
            for decision in second.admission_decisions
            if decision.reason.value == "reject_queue_full"
        ]

        duplicate_queued_decisions = [
            decision
            for decision in second.admission_decisions
            if decision.reason.value == "reject_duplicate_queued"
        ]

        second_ids = {
            task.expert_id
            for task in second.submitted_tasks
        }

        print()
        print("7-J-3 Real Qwen -> Queue Full Suppression")
        print("-------------------------------------------")
        print(f"first current experts:       {len(selected_first)}")
        print(f"second current experts:      {len(selected_second)}")
        print(f"first candidate:             {first_candidate}")
        print(f"second candidate:            {second_candidate}")
        print(f"first scheduled:             {len(first.scheduled_requests)}")
        print(f"first submitted:             {len(first.submitted_tasks)}")
        print(f"first queued IDs:            {len(first_ids)}")
        print(f"engine queue full:            {engine.queue_full}")
        print(f"second scheduled:            {len(second.scheduled_requests)}")
        print(f"second submitted:            {len(second.submitted_tasks)}")
        print(f"queue-full:                  {len(queue_full_decisions)}")
        print(f"duplicate-queued:            {len(duplicate_queued_decisions)}")
        print(f"second submitted IDs:        {len(second_ids)}")

        assert second.scheduled_requests
        assert queue_full_decisions
        assert not duplicate_queued_decisions
        assert not second_ids
        assert not (first_ids & second_ids)

        # The original queued task must still be present.
        for expert_id in first_ids:
            assert engine.task_status(expert_id) == "queued"

        # ---------------------------------------------------------
        # Release the worker and verify the original task completes.
        # ---------------------------------------------------------
        blocker_release.set()

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if all(
                engine.task_status(expert_id) == "completed"
                for expert_id in first_ids
            ):
                break
            time.sleep(0.01)

        assert engine.task_status(blocker_id) == "completed"

        for expert_id in first_ids:
            assert engine.task_status(expert_id) == "completed"

        assert set(executed_ids) >= first_ids

    finally:
        blocker_release.set()
        if engine.running:
            engine.stop(wait=True)
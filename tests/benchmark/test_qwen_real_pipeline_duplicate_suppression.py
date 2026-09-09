from __future__ import annotations

import threading
import time
import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from predictive_cache.cache import PredictiveExpertCache
from predictive_cache.prefetch.engine import PrefetchEngine
from predictive_cache.prefetch.pipeline import PrefetchPipeline
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
        # Establish real routing history and demand residency.
        # ---------------------------------------------------------
        for step in range(4):
            current = _capture_routing(
                model,
                capture,
                inputs[step % len(inputs)],
                step=step,
            )

            cache.observe(current)

        assert current

        # ---------------------------------------------------------
        # First submission.
        # ---------------------------------------------------------
        first = pipeline.process(current)

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
        second = pipeline.process(current)

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
        print(f"current experts:        {len(current)}")
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




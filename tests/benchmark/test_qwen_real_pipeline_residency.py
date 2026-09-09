from __future__ import annotations

import time

import torch

from predictive_cache.cache import PredictiveExpertCache
from predictive_cache.prefetch.engine import PrefetchEngine
from predictive_cache.prefetch.pipeline import PrefetchPipeline
from predictive_cache.routing import QwenMoeRoutingCapture
from predictive_cache.routing import QwenRoutingBridge
from predictive_cache.scheduler import ExpertScheduler
from predictive_cache.types import CacheConfig
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeForCausalLM,
)


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


def test_real_qwen_pipeline_prefetch_residency_100_steps() -> None:
    """
    7-I-1:

        real Qwen routing
            -> cache observation
            -> prediction
            -> scheduler
            -> PrefetchPipeline
            -> AdmissionController
            -> PrefetchEngine
            -> cache residency
            -> next real Qwen demand

    This benchmark deliberately validates the closed loop rather than
    re-testing synthetic expert traces from 7-G.

    Required invariants:

    1. Real Qwen routing produces experts.
    2. Scheduler requests are prediction-derived.
    3. Submitted tasks complete through the real PrefetchEngine.
    4. Completed prefetches become cache-resident.
    5. A prefetched expert used by the next real routing step is counted
       as useful.
    """

    model = _build_real_qwen()

    cache = PredictiveExpertCache(
        CacheConfig(
            capacity=2,
            prediction_top_k=1,
        )
    )

    bridge = QwenRoutingBridge(cache)
    capture = QwenMoeRoutingCapture(bridge)

    scheduler = ExpertScheduler(cache)

    total_prefetches = 0
    useful_prefetches = 0
    completed_prefetches = 0

    pending_prefetched: set[int] = set()

    def loader(task) -> None:
        nonlocal total_prefetches
        nonlocal completed_prefetches

        evicted = cache.insert(
            task.expert_id,
            size_bytes=1024,
            location="memory",
        )

        total_prefetches += 1
        completed_prefetches += 1

        pending_prefetched.add(task.expert_id)

        if evicted is not None:
            pending_prefetched.discard(evicted)

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

    routing_steps = 0
    scheduled_requests = 0
    submitted_tasks = 0
    resident_prefetched = 0

    try:
        # ---------------------------------------------------------
        # Warmup:
        #
        # Establish real Qwen routing transitions before measuring
        # predictive usefulness.
        # ---------------------------------------------------------
        for step in range(4):
            current = _capture_routing(
                model,
                capture,
                inputs[step % len(inputs)],
                step=step,
            )

            routing_steps += 1

            cache.observe(current)

            for expert_id in current:
                if cache.get(expert_id) is None:
                    cache.insert(
                        expert_id,
                        size_bytes=1024,
                        location="memory",
                    )

        # ---------------------------------------------------------
        # Measure:
        #
        # current demand first
        #     ->
        # predictor/scheduler
        #     ->
        # async prefetch
        #     ->
        # next real demand
        #
        # This ordering is important. It prevents a prefetch from
        # evicting the current demand and manufacturing a false miss.
        # ---------------------------------------------------------
        for step in range(4, 104):
            current = _capture_routing(
                model,
                capture,
                inputs[step % len(inputs)],
                step=step,
            )

            routing_steps += 1

            # Current demand is always handled first.
            cache.observe(current)

            for expert_id in current:
                if cache.get(expert_id) is None:
                    cache.insert(
                        expert_id,
                        size_bytes=1024,
                        location="memory",
                    )

                # A pending prefetch that directly satisfies the
                # current real demand is useful.
                if expert_id in pending_prefetched:
                    useful_prefetches += 1
                    pending_prefetched.discard(expert_id)

            # -----------------------------------------------------
            # Now predict what comes after the current routing.
            # -----------------------------------------------------
            result = pipeline.process(current)

            scheduled_requests += len(
                result.scheduled_requests
            )
            submitted_tasks += len(
                result.submitted_tasks
            )

            submitted_ids = [
                task.expert_id
                for task in result.submitted_tasks
            ]

            if submitted_ids:
                _wait_for_tasks(
                    engine,
                    submitted_ids,
                )

            # Every submitted task must now be genuinely resident.
            for expert_id in submitted_ids:
                if cache.contains(expert_id):
                    resident_prefetched += 1

        engine.stop(wait=True)

    finally:
        if engine.running:
            engine.stop(wait=True)

    print()
    print("7-I-2 Real Qwen -> Pipeline -> Residency Stability")
    print("-----------------------------------------")
    print(f"routing steps:          {routing_steps}")
    print(f"scheduled requests:     {scheduled_requests}")
    print(f"submitted tasks:        {submitted_tasks}")
    print(f"completed prefetches:   {completed_prefetches}")
    print(f"resident prefetches:    {resident_prefetched}")
    print(f"useful prefetches:     {useful_prefetches}")

    # -------------------------------------------------------------
    # Gate 1: the real Qwen routing path executed.
    # -------------------------------------------------------------
    assert routing_steps > 0

    # -------------------------------------------------------------
    # Gate 2: prediction actually reached Scheduler/Pipeline.
    # -------------------------------------------------------------
    assert scheduled_requests > 0
    assert submitted_tasks > 0

    # -------------------------------------------------------------
    # Gate 3: asynchronous Engine execution completed.
    # -------------------------------------------------------------
    assert completed_prefetches == submitted_tasks

    # -------------------------------------------------------------
    # Gate 4: completed prefetches became real cache residency.
    # -------------------------------------------------------------
    assert resident_prefetched == submitted_tasks

    # -------------------------------------------------------------
    # Gate 5: at least one prefetched expert eventually served
    # a real Qwen demand.
    # -------------------------------------------------------------
    assert useful_prefetches > 0

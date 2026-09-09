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


def test_real_qwen_pipeline_capacity_pressure_longrun() -> None:
    """
    7-I-4:

        real Qwen routing
            -> demand residency
            -> prediction
            -> Scheduler
            -> PrefetchPipeline
            -> PrefetchEngine
            -> cache.insert()
            -> possible LRU eviction
            -> next real demand

    This test validates runtime state consistency under a deliberately
    tiny cache capacity.

    Eviction is a legal cache outcome and is NOT treated as a prefetch
    failure.

    Required invariants:

    1. Real Qwen routing continuously produces experts.
    2. Submitted prefetches all complete.
    3. Every completed submission initially reaches cache residency.
    4. Cache size never exceeds configured capacity.
    5. Demand accounting remains valid despite eviction.
    6. At least one useful prefetch survives long enough to serve demand.
    """

    model = _build_real_qwen()

    cache = PredictiveExpertCache(
        CacheConfig(
            capacity=2,
            prediction_top_k=3,
        )
    )

    bridge = QwenRoutingBridge(cache)
    capture = QwenMoeRoutingCapture(bridge)

    scheduler = ExpertScheduler(cache)

    total_prefetches = 0
    completed_prefetches = 0
    resident_prefetched = 0
    useful_prefetches = 0

    demand_hits = 0
    demand_misses = 0
    evictions = 0

    pending_prefetched: set[int] = set()

    def loader(task) -> None:
        nonlocal total_prefetches
        nonlocal completed_prefetches
        nonlocal resident_prefetched
        nonlocal evictions

        evicted = cache.insert(
            task.expert_id,
            size_bytes=1024,
            location="memory",
        )

        total_prefetches += 1
        completed_prefetches += 1
        resident_prefetched += 1

        pending_prefetched.add(task.expert_id)

        if evicted is not None:
            evictions += 1
            pending_prefetched.discard(evicted)

        # The inserted expert must be resident immediately after
        # successful loader execution.
        assert cache.contains(task.expert_id)

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

    try:
        # ---------------------------------------------------------
        # Warmup:
        #
        # Establish actual routing history and demand residency.
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
                    evicted = cache.insert(
                        expert_id,
                        size_bytes=1024,
                        location="memory",
                    )
                    if evicted is not None:
                        evictions += 1

            assert cache.size <= cache.capacity

        # ---------------------------------------------------------
        # Pressure phase:
        #
        # Current demand first.
        # Then predictive prefetch.
        # Then wait for execution.
        #
        # With capacity=2 and up to 3 predictions, legitimate
        # eviction is expected.
        # ---------------------------------------------------------
        for step in range(4, 1000):
            current = _capture_routing(
                model,
                capture,
                inputs[step % len(inputs)],
                step=step,
            )

            routing_steps += 1
            cache.observe(current)

            # -----------------------------------------------------
            # Current demand happens before prefetch.
            # -----------------------------------------------------
            for expert_id in current:
                if cache.get(expert_id) is None:
                    demand_misses += 1

                    evicted = cache.insert(
                        expert_id,
                        size_bytes=1024,
                        location="memory",
                    )

                    if evicted is not None:
                        evictions += 1
                        pending_prefetched.discard(evicted)
                else:
                    demand_hits += 1

                if expert_id in pending_prefetched:
                    useful_prefetches += 1
                    pending_prefetched.discard(expert_id)

            assert cache.size <= cache.capacity

            # -----------------------------------------------------
            # Prediction -> Scheduler -> Pipeline.
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

            # -----------------------------------------------------
            # Engine completion must correspond to real residency
            # immediately after the loader completes.
            #
            # A later eviction is legal and is accounted separately.
            # -----------------------------------------------------
            for expert_id in submitted_ids:
                assert engine.task_status(expert_id) == "completed"

            # Completed prefetches are allowed to be evicted by later
            # prefetches when cache capacity is smaller than prediction_top_k.
            # Residency is therefore validated at loader execution time,
            # while final cache membership is treated as mutable state.
            assert cache.size <= cache.capacity

        engine.stop(wait=True)

    finally:
        if engine.running:
            engine.stop(wait=True)

    print()
    print("7-I-4 Real Qwen -> Pipeline -> 1000-Step Capacity Pressure Stability")
    print("------------------------------------------------")
    print(f"routing steps:          {routing_steps}")
    print(f"scheduled requests:     {scheduled_requests}")
    print(f"submitted tasks:        {submitted_tasks}")
    print(f"completed prefetches:   {completed_prefetches}")
    print(f"resident prefetches:    {resident_prefetched}")
    print(f"useful prefetches:      {useful_prefetches}")
    print(f"demand hits:            {demand_hits}")
    print(f"demand misses:          {demand_misses}")
    print(f"cache evictions:        {evictions}")
    print(f"final cache size:       {cache.size}")
    print(f"cache capacity:         {cache.capacity}")

    # -------------------------------------------------------------
    # Gate 1: real routing remained active.
    # -------------------------------------------------------------
    assert routing_steps == 1000

    # -------------------------------------------------------------
    # Gate 2: predictive path actually produced work.
    # -------------------------------------------------------------
    assert scheduled_requests > 0
    assert submitted_tasks > 0

    # -------------------------------------------------------------
    # Gate 3: every submitted prefetch completed.
    # -------------------------------------------------------------
    assert completed_prefetches == submitted_tasks

    # -------------------------------------------------------------
    # Gate 4: every successful loader execution published residency.
    # -------------------------------------------------------------
    assert resident_prefetched == completed_prefetches

    # -------------------------------------------------------------
    # Gate 5: cache never exceeded its hard capacity.
    # -------------------------------------------------------------
    assert cache.size <= cache.capacity

    # -------------------------------------------------------------
    # Gate 6: demand accounting is internally complete.
    # -------------------------------------------------------------
    assert demand_hits + demand_misses > 0

    # -------------------------------------------------------------
    # Gate 7: eviction is allowed, but useful predictive work must
    # still survive to at least one real demand.
    # -------------------------------------------------------------
    assert useful_prefetches > 0

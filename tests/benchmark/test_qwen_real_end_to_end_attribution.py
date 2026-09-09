from __future__ import annotations

import time

import pytest
import torch
from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

from predictive_cache import CacheConfig, PredictiveExpertCache
from predictive_cache.prefetch.engine import PrefetchEngine
from predictive_cache.prefetch.pipeline import PrefetchPipeline
from predictive_cache.prefetch.types import (
    PrefetchStatus,
    PrefetchTask,
)
from predictive_cache.routing import (
    QwenMoeRoutingCapture,
    QwenRoutingBridge,
)
from predictive_cache.scheduler import ExpertScheduler


def _build_real_qwen() -> Qwen2MoeForCausalLM:
    config = Qwen2MoeConfig(
        num_hidden_layers=2,
        hidden_size=128,
        intermediate_size=256,
        num_local_experts=8,
        num_experts_per_tok=2,
        vocab_size=128,
        hidden_act="silu",
    )

    model = Qwen2MoeForCausalLM(config)
    model.eval()
    return model


def _capture_current(
    capture: QwenMoeRoutingCapture,
    model: Qwen2MoeForCausalLM,
    input_ids: torch.Tensor,
    step: int,
) -> list[int]:
    before = len(capture.bridge.collector.events())

    with torch.no_grad():
        capture.capture_forward(
            model,
            input_ids=input_ids,
            step=step,
        )

    events = capture.bridge.collector.events()[before:]

    current: list[int] = []

    for event in events:
        if event.step != step:
            continue

        current.extend(
            int(expert_id)
            for expert_id in event.expert_ids
        )

    return sorted(set(current))


def _wait_until_terminal(
    pipeline: PrefetchPipeline,
    expected: int,
    timeout: float = 10.0,
) -> list:
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        results = pipeline.results()

        terminal = [
            result
            for result in results
            if result.status
            in {
                PrefetchStatus.COMPLETED,
                PrefetchStatus.FAILED,
                PrefetchStatus.CANCELLED,
            }
        ]

        if len(terminal) >= expected:
            return terminal

        time.sleep(0.01)

    results = pipeline.results()

    terminal = [
        result
        for result in results
        if result.status
        in {
            PrefetchStatus.COMPLETED,
            PrefetchStatus.FAILED,
            PrefetchStatus.CANCELLED,
        }
    ]

    return terminal


def test_real_qwen_end_to_end_prefetch_attribution() -> None:
    """
    7-L-3:

        REAL Qwen routing
            -> prediction
            -> admission
            -> submission
            -> completion
            -> residency
            -> later demand
            -> useful / wasted attribution

    Attribution is based only on tasks that were actually admitted
    and submitted to the prefetch engine.

    No production code is modified.
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

    load_attempts: dict[int, int] = {}

    def loader(task: PrefetchTask) -> None:
        expert_id = int(task.expert_id)

        load_attempts[expert_id] = (
            load_attempts.get(expert_id, 0) + 1
        )

        cache.insert(
            expert_id,
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

    inputs = [
        torch.tensor(
            [[10 + (step % 118)]],
            dtype=torch.long,
        )
        for step in range(32)
    ]

    try:
        routed: list[list[int]] = []

        # ---------------------------------------------------------
        # Phase 1:
        # Establish real Qwen routing history.
        # ---------------------------------------------------------
        for step, input_ids in enumerate(inputs[:8]):
            current = _capture_current(
                capture,
                model,
                input_ids,
                step,
            )

            assert current, (
                f"Qwen routing produced no experts at step {step}"
            )

            routed.append(current)

        # ---------------------------------------------------------
        # Phase 2:
        # Find a real prediction/demand pair.
        #
        # We deliberately search the observed routing trace rather
        # than hard-coding expert IDs.
        # ---------------------------------------------------------
        selected = None

        for prediction_index in range(len(routed) - 1):
            prediction_current = routed[prediction_index]

            predicted_requests = (
                scheduler.plan_prefetch_predictions(
                    prediction_current
                )
            )

            predicted_ids = {
                int(request.expert_id)
                for request in predicted_requests
            }

            if not predicted_ids:
                continue

            for demand_index in range(
                prediction_index + 1,
                len(routed),
            ):
                demand_current = set(routed[demand_index])

                useful_candidates = (
                    predicted_ids & demand_current
                )
                wasted_candidates = (
                    predicted_ids - demand_current
                )

                if useful_candidates:
                    selected = (
                        prediction_index,
                        prediction_current,
                        predicted_requests,
                        demand_index,
                        demand_current,
                        useful_candidates,
                        wasted_candidates,
                    )

                    # Prefer a pair that contains both useful
                    # and wasted candidates, but do not require
                    # that outcome from a short stochastic trace.
                    if wasted_candidates:
                        break

            if selected is not None:
                break

        assert selected is not None, (
            "Could not find a real Qwen routing pair with "
            "both useful and wasted prediction candidates"
        )

        (
            prediction_index,
            prediction_current,
            predicted_requests,
            demand_index,
            demand_current,
            expected_useful,
            expected_wasted,
        ) = selected

        predicted_ids = {
            int(request.expert_id)
            for request in predicted_requests
        }

        # ---------------------------------------------------------
        # Phase 3:
        # Run the ACTUAL pipeline.
        #
        # This is important: attribution must use the pipeline's
        # admission decisions, not merely scheduler output.
        # ---------------------------------------------------------
        result = pipeline.process(
            prediction_current,
        )

        accepted_tasks = [
            task
            for task, decision in zip(
                result.submitted_tasks,
                [
                    decision
                    for decision in result.admission_decisions
                    if decision.admitted
                ],
            )
        ]

        # submitted_tasks are exactly the tasks which survived
        # admission and were submitted to the engine.
        admitted_ids = {
            int(task.expert_id)
            for task in result.submitted_tasks
        }

        accepted_decisions = [
            decision
            for decision in result.admission_decisions
            if decision.admitted
        ]

        assert len(accepted_decisions) == len(
            result.submitted_tasks
        )

        assert admitted_ids == {
            int(task.expert_id)
            for task in result.submitted_tasks
        }

        assert admitted_ids <= predicted_ids

        # ---------------------------------------------------------
        # Phase 4:
        # Wait for every admitted prefetch to reach terminal state.
        # ---------------------------------------------------------
        terminal_results = _wait_until_terminal(
            pipeline,
            expected=len(result.submitted_tasks),
        )

        assert len(terminal_results) == len(
            result.submitted_tasks
        )

        completed_ids = {
            int(prefetch_result.expert_id)
            for prefetch_result in terminal_results
            if prefetch_result.status
            == PrefetchStatus.COMPLETED
        }

        failed_ids = {
            int(prefetch_result.expert_id)
            for prefetch_result in terminal_results
            if prefetch_result.status
            == PrefetchStatus.FAILED
        }

        cancelled_ids = {
            int(prefetch_result.expert_id)
            for prefetch_result in terminal_results
            if prefetch_result.status
            == PrefetchStatus.CANCELLED
        }

        assert not failed_ids
        assert not cancelled_ids

        assert completed_ids == admitted_ids

        # ---------------------------------------------------------
        # Phase 5:
        # Residency must be established by the actual loader.
        # ---------------------------------------------------------
        resident_prefetched = {
            expert_id
            for expert_id in completed_ids
            if cache.contains(expert_id)
        }

        assert resident_prefetched == completed_ids

        # ---------------------------------------------------------
        # Phase 6:
        # Perform the later DEMAND through the actual cache API.
        #
        # We intentionally do not merely inspect membership here.
        # cache.get() is the real demand path and therefore updates
        # hit/miss accounting.
        # ---------------------------------------------------------
        demand_hits: set[int] = set()
        demand_misses: set[int] = set()

        for expert_id in demand_current:
            value = cache.get(expert_id)

            if value is None:
                demand_misses.add(expert_id)
            else:
                demand_hits.add(expert_id)

        # ---------------------------------------------------------
        # Phase 7:
        # Attribution is restricted to admitted + completed
        # prefetches.
        # ---------------------------------------------------------
        useful_prefetches = (
            completed_ids & demand_hits
        )

        wasted_prefetches = (
            completed_ids - demand_hits
        )

        # Every completed prefetch must belong to exactly one
        # attribution bucket.
        assert useful_prefetches.isdisjoint(
            wasted_prefetches
        )

        assert (
            len(useful_prefetches)
            + len(wasted_prefetches)
            == len(completed_ids)
        )

        # Gate:
        # We need both sides of the attribution to be observable.
        assert useful_prefetches, (
            "7-L-3 requires at least one useful admitted prefetch"
        )

        assert wasted_prefetches, (
            "7-L-3 requires at least one wasted admitted prefetch"
        )

        # The actual demand relation must agree with attribution.
        expected_useful = (
            completed_ids & demand_current
        )

        expected_wasted = (
            completed_ids - demand_current
        )

        assert useful_prefetches == expected_useful
        assert wasted_prefetches == expected_wasted

        # ---------------------------------------------------------
        # Phase 8:
        # Final accounting invariants.
        # ---------------------------------------------------------
        assert (
            len(result.scheduled_requests)
            >= len(result.submitted_tasks)
        )

        assert (
            result.admission_stats.accepted
            == len(result.submitted_tasks)
        )

        assert (
            result.admission_stats.total
            == len(result.admission_decisions)
        )

        assert (
            result.admission_stats.accepted
            + result.admission_stats.rejected
            == result.admission_stats.total
        )

        assert (
            len(completed_ids)
            + len(failed_ids)
            + len(cancelled_ids)
            == len(result.submitted_tasks)
        )

        assert pipeline.queue_size == 0

        # ---------------------------------------------------------
        # Report.
        # ---------------------------------------------------------
        print()
        print("7-L-3 Real Qwen -> End-to-End Attribution")
        print("------------------------------------------")
        print(
            f"prediction step:          {prediction_index}"
        )
        print(
            f"later demand step:        {demand_index}"
        )
        print(
            f"prediction current:       "
            f"{len(prediction_current)}"
        )
        print(
            f"predicted candidates:     "
            f"{len(predicted_ids)}"
        )
        print(
            f"scheduled requests:       "
            f"{len(result.scheduled_requests)}"
        )
        print(
            f"admission decisions:      "
            f"{len(result.admission_decisions)}"
        )
        print(
            f"admitted/submitted:       "
            f"{len(result.submitted_tasks)}"
        )
        print(
            f"completed:                "
            f"{len(completed_ids)}"
        )
        print(
            f"failed:                   "
            f"{len(failed_ids)}"
        )
        print(
            f"cancelled:                "
            f"{len(cancelled_ids)}"
        )
        print(
            f"later demand experts:     "
            f"{len(demand_current)}"
        )
        print(
            f"demand hits:              "
            f"{len(demand_hits)}"
        )
        print(
            f"demand misses:            "
            f"{len(demand_misses)}"
        )
        print(
            f"useful admitted prefetch: "
            f"{len(useful_prefetches)}"
        )
        print(
            f"wasted admitted prefetch: "
            f"{len(wasted_prefetches)}"
        )
        print(
            f"useful expert IDs:        "
            f"{sorted(useful_prefetches)}"
        )
        print(
            f"wasted expert IDs:        "
            f"{sorted(wasted_prefetches)}"
        )
        print(
            f"resident completed:       "
            f"{len(resident_prefetched)}"
        )
        print(
            f"admission accepted:       "
            f"{result.admission_stats.accepted}"
        )
        print(
            f"admission rejected:       "
            f"{result.admission_stats.rejected}"
        )
        print(
            f"queue size:               "
            f"{pipeline.queue_size}"
        )

    finally:
        pipeline.stop(wait=True)
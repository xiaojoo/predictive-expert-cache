from __future__ import annotations

import time

import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from predictive_cache.cache import PredictiveExpertCache
from predictive_cache.prefetch.engine import PrefetchEngine
from predictive_cache.prefetch.pipeline import PrefetchPipeline
from predictive_cache.prefetch.types import PrefetchStatus, PrefetchTask
from predictive_cache.routing import QwenMoeRoutingCapture, QwenRoutingBridge
from predictive_cache.scheduler import ExpertScheduler
from predictive_cache.types import CacheConfig


def _build_real_qwen() -> Qwen3MoeForCausalLM:
    model = Qwen3MoeForCausalLM(
        Qwen3MoeConfig(
            num_hidden_layers=2,
            hidden_size=128,
            intermediate_size=256,
            num_local_experts=8,
            num_experts_per_tok=2,
            vocab_size=128,
        )
    )
    model.eval()
    return model


def _capture_current(
    capture: QwenMoeRoutingCapture,
    model: Qwen3MoeForCausalLM,
    input_ids: torch.Tensor,
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

    current = sorted(
        {
            expert_id
            for event in events
            if event.step == step
            for expert_id in event.expert_ids
        }
    )

    assert current
    return current


def _wait_completed(
    engine: PrefetchEngine,
    expert_ids: set[int],
    timeout: float = 5.0,
) -> None:
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if all(
            engine.task_status(expert_id)
            == PrefetchStatus.COMPLETED
            for expert_id in expert_ids
        ):
            return

        time.sleep(0.01)

    raise AssertionError(
        "prefetch tasks did not complete: "
        f"{sorted(expert_ids)}"
    )


def test_real_qwen_useful_and_wasted_prefetch_attribution() -> None:
    """
    7-L-2:

        Real Qwen routing
            -> prediction
            -> prefetch
            -> residency
            -> later demand
            -> useful / wasted attribution

    A prefetch is considered useful only when a later demand
    actually hits the prefetched expert in cache.

    A completed resident prefetch that is not demanded by the
    selected later routing event is considered wasted for this
    benchmark.

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

    attempts: dict[int, int] = {}

    def loader(task: PrefetchTask) -> None:
        expert_id = int(task.expert_id)

        attempts[expert_id] = (
            attempts.get(expert_id, 0) + 1
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
        torch.tensor([[10, 11, 12]], dtype=torch.long),
        torch.tensor([[20, 21, 22]], dtype=torch.long),
        torch.tensor([[30, 31, 32]], dtype=torch.long),
        torch.tensor([[40, 41, 42]], dtype=torch.long),
        torch.tensor([[50, 51, 52]], dtype=torch.long),
        torch.tensor([[60, 61, 62]], dtype=torch.long),
        torch.tensor([[70, 71, 72]], dtype=torch.long),
        torch.tensor([[80, 81, 82]], dtype=torch.long),
    ]

    try:
        routed: list[list[int]] = []

        # ---------------------------------------------------------
        # Phase 1:
        # Establish REAL Qwen routing history.
        # ---------------------------------------------------------
        for step, input_ids in enumerate(inputs):
            current = _capture_current(
                capture,
                model,
                input_ids,
                step,
            )

            cache.observe(current)

            routed.append(current)

        assert routed

        # ---------------------------------------------------------
        # Phase 2:
        # Find a REAL prediction/demand pair.
        #
        # Requirements:
        #
        #   prediction state
        #       -> >= 2 predicted candidates
        #
        #   later demand state
        #       -> contains at least one predicted candidate
        #       -> excludes at least one predicted candidate
        #
        # This creates deterministic useful/wasted attribution
        # without hardcoding expert IDs.
        # ---------------------------------------------------------
        selected_current: list[int] | None = None
        selected_requests = []
        demand_current: list[int] | None = None

        for prediction_index, candidate_current in enumerate(
            routed
        ):
            requests = scheduler.plan_prefetch_predictions(
                candidate_current
            )

            if len(requests) < 2:
                continue

            candidate_ids = {
                int(request.expert_id)
                for request in requests
            }

            for later_current in routed[
                prediction_index + 1 :
            ]:
                overlap = candidate_ids.intersection(
                    later_current
                )

                absent = candidate_ids.difference(
                    later_current
                )

                if overlap and absent:
                    selected_current = candidate_current
                    selected_requests = requests
                    demand_current = later_current
                    break

            if selected_current is not None:
                break

        assert selected_current is not None
        assert len(selected_requests) >= 2
        assert demand_current is not None

        predicted_ids = {
            int(request.expert_id)
            for request in selected_requests
        }

        demanded_predicted_ids = predicted_ids.intersection(
            demand_current
        )

        wasted_predicted_ids = predicted_ids.difference(
            demand_current
        )

        assert demanded_predicted_ids
        assert wasted_predicted_ids

        # ---------------------------------------------------------
        # Phase 3:
        # Execute the actual prefetch pipeline.
        # ---------------------------------------------------------
        first = pipeline.process(
            selected_current
        )

        assert first.scheduled_requests
        assert first.submitted_tasks

        prefetched_ids = {
            int(task.expert_id)
            for task in first.submitted_tasks
        }

        assert prefetched_ids

        # Scheduler/admission may filter predictions, so attribution
        # is always based on actually submitted tasks.
        useful_candidates = prefetched_ids.intersection(
            demand_current
        )

        wasted_candidates = prefetched_ids.difference(
            demand_current
        )

        assert useful_candidates
        assert wasted_candidates

        # ---------------------------------------------------------
        # Phase 4:
        # Wait until every actual prefetch completes.
        # ---------------------------------------------------------
        _wait_completed(
            engine,
            prefetched_ids,
        )

        assert all(
            engine.task_status(expert_id)
            == PrefetchStatus.COMPLETED
            for expert_id in prefetched_ids
        )

        # Every completed prefetch must become resident because
        # the synthetic loader explicitly inserts it into RAM.
        assert all(
            cache.contains(expert_id)
            for expert_id in prefetched_ids
        )

        # ---------------------------------------------------------
        # Phase 5:
        # Generate the actual demand event.
        #
        # The demand state itself came from REAL Qwen routing.
        # cache.get() is the actual cache-demand operation.
        # ---------------------------------------------------------
        demand_hits: set[int] = set()

        for expert_id in demand_current:
            entry = cache.get(
                expert_id
            )

            if entry is not None:
                demand_hits.add(
                    int(expert_id)
                )

        # ---------------------------------------------------------
        # Phase 6:
        # Attribute usefulness.
        #
        # Useful:
        #     actually prefetched
        #     AND subsequently demanded
        #     AND cache.get() hit
        #
        # Wasted:
        #     actually prefetched
        #     BUT not demanded by this demand event
        # ---------------------------------------------------------
        useful_prefetches = (
            prefetched_ids.intersection(
                demand_hits
            )
        )

        wasted_prefetches = (
            prefetched_ids.difference(
                demand_hits
            )
        )

        assert useful_prefetches
        assert wasted_prefetches

        assert useful_prefetches.isdisjoint(
            wasted_prefetches
        )

        # The selected routing pair predicted at least one useful
        # and one non-demanded candidate. The actual submitted set
        # must preserve that distinction.
        assert useful_prefetches == useful_candidates
        assert wasted_prefetches == wasted_candidates

        # ---------------------------------------------------------
        # Phase 7:
        # Final residency/accounting invariants.
        # ---------------------------------------------------------
        assert all(
            cache.contains(expert_id)
            for expert_id in useful_prefetches
        )

        assert all(
            cache.contains(expert_id)
            for expert_id in wasted_prefetches
        )

        assert (
            len(useful_prefetches)
            + len(wasted_prefetches)
            == len(prefetched_ids)
        )

        # Every submitted task executed exactly once.
        assert all(
            attempts.get(expert_id) == 1
            for expert_id in prefetched_ids
        )

        assert engine.unfinished_tasks == 0

        print()
        print(
            "7-L-2 Real Qwen -> "
            "Useful / Wasted Attribution"
        )
        print(
            "-----------------------------------------------"
        )
        print(
            f"prediction current experts: "
            f"{len(selected_current)}"
        )
        print(
            f"predicted candidates:        "
            f"{len(predicted_ids)}"
        )
        print(
            f"scheduled requests:          "
            f"{len(first.scheduled_requests)}"
        )
        print(
            f"submitted prefetches:        "
            f"{len(prefetched_ids)}"
        )
        print(
            f"later demand experts:        "
            f"{len(demand_current)}"
        )
        print(
            f"useful prefetches:           "
            f"{len(useful_prefetches)}"
        )
        print(
            f"wasted prefetches:           "
            f"{len(wasted_prefetches)}"
        )
        print(
            f"useful expert IDs:           "
            f"{sorted(useful_prefetches)}"
        )
        print(
            f"wasted expert IDs:           "
            f"{sorted(wasted_prefetches)}"
        )
        print(
            f"resident prefetched:         "
            f"{sum(cache.contains(i) for i in prefetched_ids)}"
        )
        print(
            f"unfinished tasks:             "
            f"{engine.unfinished_tasks}"
        )

    finally:
        if engine.running:
            engine.stop(wait=True)

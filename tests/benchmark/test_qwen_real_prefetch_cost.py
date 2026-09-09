from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass

import torch
from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

from predictive_cache import CacheConfig, PredictiveExpertCache
from predictive_cache.prefetch.engine import PrefetchEngine
from predictive_cache.prefetch.pipeline import PrefetchPipeline
from predictive_cache.prefetch.types import PrefetchStatus, PrefetchTask
from predictive_cache.routing import QwenMoeRoutingCapture, QwenRoutingBridge
from predictive_cache.scheduler import ExpertScheduler


@dataclass
class LoadStats:
    calls: int = 0
    loaded_experts: list[int] | None = None

    def __post_init__(self) -> None:
        if self.loaded_experts is None:
            self.loaded_experts = []


def _build_real_qwen() -> Qwen2MoeForCausalLM:
    config = Qwen2MoeConfig(
        num_hidden_layers=2,
        hidden_size=128,
        intermediate_size=256,
        num_local_experts=8,
        num_experts_per_tok=2,
        vocab_size=128,
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


def _wait_terminal(
    engine: PrefetchEngine,
    tasks: list[PrefetchTask],
    timeout: float = 5.0,
) -> Counter[PrefetchStatus]:
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        statuses = [
            engine.task_status(task.expert_id)
            for task in tasks
        ]

        if all(
            status
            in {
                PrefetchStatus.COMPLETED,
                PrefetchStatus.FAILED,
                PrefetchStatus.CANCELLED,
            }
            for status in statuses
        ):
            return Counter(
                status
                for status in statuses
                if status is not None
            )

        time.sleep(0.005)

    raise AssertionError(
        "Prefetch tasks did not reach terminal state: "
        f"{[(task.expert_id, engine.task_status(task.expert_id)) for task in tasks]}"
    )


def _build_pipeline(
    cache: PredictiveExpertCache,
    loader,
) -> tuple[ExpertScheduler, PrefetchEngine, PrefetchPipeline]:
    scheduler = ExpertScheduler(cache)

    engine = PrefetchEngine(
        loader=loader,
        num_workers=2,
        max_queue_size=16,
    )

    pipeline = PrefetchPipeline(
        scheduler,
        engine,
        auto_start=True,
    )

    return scheduler, engine, pipeline


def _warm_qwen(
    capture: QwenMoeRoutingCapture,
    model: Qwen2MoeForCausalLM,
    cache: PredictiveExpertCache,
    steps: int = 32,
) -> list[list[int]]:
    routed: list[list[int]] = []

    for step in range(steps):
        input_ids = torch.tensor(
            [[10 + (step % 118)]],
            dtype=torch.long,
        )

        current = _capture_current(
            capture,
            model,
            input_ids,
            step,
        )

        assert current
        routed.append(current)

    assert any(
        cache.prefetch_candidates(current)
        for current in routed
    ), "Real Qwen produced no prefetch candidates"

    return routed


def _make_loader(
    cache: PredictiveExpertCache,
    stats: LoadStats,
):
    def loader(task: PrefetchTask) -> None:
        expert_id = int(task.expert_id)

        stats.calls += 1
        stats.loaded_experts.append(expert_id)

        cache.insert(
            expert_id,
            size_bytes=1024,
            location="memory",
        )

    return loader


def test_real_qwen_cold_vs_prefetched_load_path() -> None:
    print()
    print("7-N-1 Real Qwen -> Cold vs Prefetched Load Path")
    print("------------------------------------------------")

    model = _build_real_qwen()

    cache = PredictiveExpertCache(
        CacheConfig(
            capacity=8,
            prediction_top_k=3,
        )
    )

    bridge = QwenRoutingBridge(cache)
    capture = QwenMoeRoutingCapture(bridge)

    stats = LoadStats()
    loader = _make_loader(cache, stats)

    scheduler, engine, pipeline = _build_pipeline(
        cache,
        loader,
    )

    try:
        routed = _warm_qwen(
            capture,
            model,
            cache,
            steps=32,
        )

        prediction_current = next(
            current
            for current in routed
            if scheduler.plan_prefetch_predictions(current)
        )

        predicted_requests = (
            scheduler.plan_prefetch_predictions(
                prediction_current
            )
        )

        assert predicted_requests

        selected = int(
            predicted_requests[0].expert_id
        )

        # The selected prediction is initially cold.
        assert cache.get(selected) is None

        before_cold = stats.calls

        # Simulate the genuine cold demand load.
        loader(
            PrefetchTask(
                expert_id=selected,
                source="nvme",
                target="memory",
                priority=0,
                confidence=1.0,
                estimated_distance=0,
                source_path="benchmark",
            )
        )

        cold_loader_calls = (
            stats.calls - before_cold
        )

        assert cold_loader_calls == 1
        assert cache.get(selected) is not None

        # Reset only residency. Predictor history remains intact.
        cache.clear()

        assert not cache.contains(selected)

        before_prefetch = stats.calls

        result = pipeline.process(
            prediction_current
        )

        submitted = [
            task
            for task in result.submitted_tasks
            if int(task.expert_id) == selected
        ]

        assert submitted, (
            "Selected real prediction was not submitted"
        )

        terminal = _wait_terminal(
            engine,
            submitted,
        )

        assert terminal[PrefetchStatus.COMPLETED] == len(
            submitted
        )

        prefetch_loader_calls = (
            stats.calls - before_prefetch
        )

        assert prefetch_loader_calls == len(
            result.submitted_tasks
        )
        assert cache.get(selected) is not None

        # Demand after successful prefetch must be a cache hit.
        before_warm_demand = stats.calls

        warm_hit = cache.get(selected)

        assert warm_hit is not None
        assert (
            stats.calls - before_warm_demand
        ) == 0

        print(
            f"prediction current:       {sorted(prediction_current)}"
        )
        print(f"selected expert:          {selected}")
        print("cold demand hit:          False")
        print(f"cold loader calls:        {cold_loader_calls}")
        print(
            f"prefetch loader calls:    {prefetch_loader_calls}"
        )
        print("prefetched resident:      True")
        print("warm demand loader delta: 0")

    finally:
        pipeline.stop()


def test_real_qwen_prefetch_reduces_demand_side_load() -> None:
    print()
    print("7-N-2 Real Qwen -> Prefetch Reduces Demand-Side Load")
    print("----------------------------------------------------")

    model = _build_real_qwen()

    cache = PredictiveExpertCache(
        CacheConfig(
            capacity=8,
            prediction_top_k=3,
        )
    )

    bridge = QwenRoutingBridge(cache)
    capture = QwenMoeRoutingCapture(bridge)

    stats = LoadStats()
    loader = _make_loader(cache, stats)

    scheduler, engine, pipeline = _build_pipeline(
        cache,
        loader,
    )

    try:
        routed = _warm_qwen(
            capture,
            model,
            cache,
            steps=32,
        )

        selected_data = None

        for prediction_index, prediction_current in enumerate(
            routed[:-1]
        ):
            predicted = (
                scheduler.plan_prefetch_predictions(
                    prediction_current
                )
            )

            predicted_ids = {
                int(request.expert_id)
                for request in predicted
            }

            if not predicted_ids:
                continue

            for demand_index in range(
                prediction_index + 1,
                len(routed),
            ):
                demand_current = set(
                    routed[demand_index]
                )

                useful = (
                    predicted_ids
                    & demand_current
                )

                if useful:
                    selected_data = (
                        prediction_index,
                        prediction_current,
                        demand_index,
                        demand_current,
                        next(iter(useful)),
                    )
                    break

            if selected_data is not None:
                break

        assert selected_data is not None, (
            "Could not find a real Qwen prediction/demand pair"
        )

        (
            prediction_index,
            prediction_current,
            demand_index,
            demand_current,
            selected,
        ) = selected_data

        # Cold state.
        cache.clear()

        assert cache.get(selected) is None

        before_cold = stats.calls

        loader(
            PrefetchTask(
                expert_id=selected,
                source="nvme",
                target="memory",
                priority=0,
                confidence=1.0,
                estimated_distance=0,
                source_path="benchmark",
            )
        )

        cold_loads = (
            stats.calls - before_cold
        )

        assert cold_loads == 1

        # Reset residency while preserving predictor history.
        cache.clear()

        assert cache.get(selected) is None

        # Predictive path.
        result = pipeline.process(
            prediction_current
        )

        submitted = [
            task
            for task in result.submitted_tasks
            if int(task.expert_id) == selected
        ]

        assert submitted

        terminal = _wait_terminal(
            engine,
            submitted,
        )

        assert terminal[PrefetchStatus.COMPLETED] == len(
            submitted
        )

        assert cache.get(selected) is not None

        # Actual later demand: no loader call is necessary.
        before_demand = stats.calls

        demand_hit = cache.get(selected)

        assert demand_hit is not None

        demand_loader_delta = (
            stats.calls - before_demand
        )

        assert demand_loader_delta == 0

        print(f"prediction step:          {prediction_index}")
        print(f"demand step:              {demand_index}")
        print(
            f"prediction current:       {sorted(prediction_current)}"
        )
        print(
            f"demand experts:           {sorted(demand_current)}"
        )
        print(f"selected expert:          {selected}")
        print(f"cold demand loads:        {cold_loads}")
        print("prefetched demand hit:    True")
        print(
            f"demand-side loader delta: {demand_loader_delta}"
        )

    finally:
        pipeline.stop()


def test_real_qwen_wasted_prefetch_cost() -> None:
    print()
    print("7-N-3 Real Qwen -> Wasted Prefetch Cost")
    print("---------------------------------------")

    model = _build_real_qwen()

    cache = PredictiveExpertCache(
        CacheConfig(
            capacity=8,
            prediction_top_k=3,
        )
    )

    bridge = QwenRoutingBridge(cache)
    capture = QwenMoeRoutingCapture(bridge)

    stats = LoadStats()
    loader = _make_loader(cache, stats)

    scheduler, engine, pipeline = _build_pipeline(
        cache,
        loader,
    )

    try:
        routed = _warm_qwen(
            capture,
            model,
            cache,
            steps=32,
        )

        selected_data = None

        for prediction_index, prediction_current in enumerate(
            routed[:-1]
        ):
            predicted = (
                scheduler.plan_prefetch_predictions(
                    prediction_current
                )
            )

            predicted_ids = {
                int(request.expert_id)
                for request in predicted
            }

            if not predicted_ids:
                continue

            for demand_index in range(
                prediction_index + 1,
                len(routed),
            ):
                demand_current = set(
                    routed[demand_index]
                )

                useful = (
                    predicted_ids
                    & demand_current
                )

                wasted = (
                    predicted_ids
                    - demand_current
                )

                if useful and wasted:
                    selected_data = (
                        prediction_index,
                        prediction_current,
                        demand_index,
                        demand_current,
                        predicted_ids,
                        useful,
                        wasted,
                    )
                    break

            if selected_data is not None:
                break

        assert selected_data is not None, (
            "Could not find real Qwen useful+wasted prediction pair"
        )

        (
            prediction_index,
            prediction_current,
            demand_index,
            demand_current,
            predicted_ids,
            useful,
            wasted,
        ) = selected_data

        # Make every predicted candidate genuinely cold.
        cache.clear()

        for expert_id in predicted_ids:
            assert not cache.contains(
                expert_id
            )

        before_loader = stats.calls

        result = pipeline.process(
            prediction_current
        )

        submitted = list(
            result.submitted_tasks
        )

        assert submitted

        terminal = _wait_terminal(
            engine,
            submitted,
        )

        completed = terminal[
            PrefetchStatus.COMPLETED
        ]
        failed = terminal[
            PrefetchStatus.FAILED
        ]
        cancelled = terminal[
            PrefetchStatus.CANCELLED
        ]

        assert (
            completed
            + failed
            + cancelled
            == len(submitted)
        )

        completed_ids = {
            int(task.expert_id)
            for task in submitted
            if engine.task_status(
                task.expert_id
            )
            == PrefetchStatus.COMPLETED
        }

        assert completed_ids

        demand_hits = {
            int(expert_id)
            for expert_id in demand_current
            if cache.get(
                int(expert_id)
            )
            is not None
        }

        useful_completed = (
            completed_ids
            & demand_hits
        )

        wasted_completed = (
            completed_ids
            - demand_current
        )

        loader_delta = (
            stats.calls - before_loader
        )

        assert useful_completed
        assert wasted_completed

        assert (
            len(useful_completed)
            + len(wasted_completed)
            == completed
        )

        print(f"prediction step:          {prediction_index}")
        print(f"demand step:              {demand_index}")
        print(
            f"predicted IDs:             {sorted(predicted_ids)}"
        )
        print(
            f"demand experts:            {sorted(demand_current)}"
        )
        print(
            f"submitted prefetches:      {len(submitted)}"
        )
        print(
            f"completed prefetches:      {completed}"
        )
        print(
            f"failed prefetches:         {failed}"
        )
        print(
            f"cancelled prefetches:      {cancelled}"
        )
        print(
            f"useful completed:          {len(useful_completed)}"
        )
        print(
            f"wasted completed:          {len(wasted_completed)}"
        )
        print(
            f"useful expert IDs:         {sorted(useful_completed)}"
        )
        print(
            f"wasted expert IDs:         {sorted(wasted_completed)}"
        )
        print(
            f"prefetch loader calls:     {loader_delta}"
        )

    finally:
        pipeline.stop()


def test_real_qwen_long_run_prefetch_roi() -> None:
    print()
    print("7-N-4 Real Qwen -> Long-Run Prefetch ROI")
    print("----------------------------------------")

    model = _build_real_qwen()

    cache = PredictiveExpertCache(
        CacheConfig(
            capacity=8,
            prediction_top_k=3,
        )
    )

    bridge = QwenRoutingBridge(cache)
    capture = QwenMoeRoutingCapture(bridge)

    stats = LoadStats()
    loader = _make_loader(cache, stats)

    scheduler, engine, pipeline = _build_pipeline(
        cache,
        loader,
    )

    try:
        _warm_qwen(
            capture,
            model,
            cache,
            steps=32,
        )

        total_steps = 100

        scheduled = 0
        submitted = 0
        completed = 0
        failed = 0
        cancelled = 0

        demand_hits = 0
        demand_misses = 0

        useful = 0
        wasted = 0

        steps_with_prediction = 0
        steps_with_useful_prefetch = 0

        # Experts whose prefetch completed before a future demand.
        #
        # expert_id -> prefetch step
        pending_prefetches: dict[int, int] = {}

        # Experts already attributed as useful.
        useful_experts: set[int] = set()

        for offset in range(total_steps):
            step = 32 + offset

            input_ids = torch.tensor(
                [[10 + (offset % 118)]],
                dtype=torch.long,
            )

            current = _capture_current(
                capture,
                model,
                input_ids,
                step,
            )

            assert current

            current_set = set(current)

            # -------------------------------------------------
            # Demand first:
            #
            # This demand represents the future use of prefetches
            # produced by earlier routing steps.
            # -------------------------------------------------

            step_useful: set[int] = set()

            for expert_id in current_set:
                if cache.get(expert_id) is not None:
                    demand_hits += 1

                    if expert_id in pending_prefetches:
                        step_useful.add(expert_id)
                        useful_experts.add(expert_id)
                else:
                    demand_misses += 1

            if step_useful:
                steps_with_useful_prefetch += 1

            useful += len(step_useful)

            # A prefetched expert that has now been consumed is
            # removed from the pending attribution set.
            for expert_id in step_useful:
                pending_prefetches.pop(
                    expert_id,
                    None,
                )

            # -------------------------------------------------
            # Now process the current routing step.
            #
            # Its predictions are for future demand.
            # -------------------------------------------------

            predicted_requests = (
                scheduler.plan_prefetch_predictions(
                    current
                )
            )

            predicted_ids = {
                int(request.expert_id)
                for request in predicted_requests
            }

            if predicted_ids:
                steps_with_prediction += 1

            result = pipeline.process(
                current
            )

            scheduled += len(
                result.scheduled_requests
            )

            submitted += len(
                result.submitted_tasks
            )

            tasks = list(
                result.submitted_tasks
            )

            if tasks:
                terminal = _wait_terminal(
                    engine,
                    tasks,
                )

                completed += terminal[
                    PrefetchStatus.COMPLETED
                ]

                failed += terminal[
                    PrefetchStatus.FAILED
                ]

                cancelled += terminal[
                    PrefetchStatus.CANCELLED
                ]

            # -------------------------------------------------
            # Completed prefetches become candidates for future
            # useful attribution.
            # -------------------------------------------------

            for task in tasks:
                expert_id = int(task.expert_id)

                if (
                    engine.task_status(expert_id)
                    == PrefetchStatus.COMPLETED
                ):
                    pending_prefetches[
                        expert_id
                    ] = step

        terminal_accounted = (
            completed
            + failed
            + cancelled
        )

        assert submitted == terminal_accounted
        assert engine.unfinished_tasks == 0

        # Every completed prefetch is either:
        #
        #   useful -> eventually consumed while resident
        #   wasted -> never consumed
        #
        # Anything still pending at the end is also wasted for the
        # finite benchmark horizon.
        still_pending = set(
            pending_prefetches
        )

        wasted = (
            completed
            - useful
        )

        assert useful + wasted == completed
        assert (
            useful_experts
            <= set(
                stats.loaded_experts
            )
        )

        useful_rate = (
            useful / completed
            if completed
            else 0.0
        )

        print("warm-up steps:             32")
        print(
            f"long-run routing steps:    {total_steps}"
        )
        print(
            f"steps with prediction:     {steps_with_prediction}"
        )
        print(
            f"steps with useful prefetch:{steps_with_useful_prefetch}"
        )
        print(
            f"scheduled requests:        {scheduled}"
        )
        print(
            f"submitted prefetches:      {submitted}"
        )
        print(
            f"completed prefetches:      {completed}"
        )
        print(
            f"failed prefetches:         {failed}"
        )
        print(
            f"cancelled prefetches:      {cancelled}"
        )
        print(
            f"demand hits:               {demand_hits}"
        )
        print(
            f"demand misses:             {demand_misses}"
        )
        print(
            f"useful prefetches:         {useful}"
        )
        print(
            f"wasted prefetches:         {wasted}"
        )
        print(
            f"still pending at end:      {len(still_pending)}"
        )
        print(
            f"useful prefetch rate:      {useful_rate:.3f}"
        )
        print(
            f"loader calls:              {stats.calls}"
        )
        print(
            f"unfinished tasks:          {engine.unfinished_tasks}"
        )

        assert steps_with_prediction > 0
        assert submitted > 0
        assert completed > 0
        assert engine.unfinished_tasks == 0

    finally:
        pipeline.stop()

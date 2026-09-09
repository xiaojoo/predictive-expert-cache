from __future__ import annotations

import time

import torch
from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

from predictive_cache import CacheConfig, PredictiveExpertCache
from predictive_cache.prefetch.engine import PrefetchEngine
from predictive_cache.prefetch.pipeline import PrefetchPipeline
from predictive_cache.prefetch.types import PrefetchTask
from predictive_cache.routing import (
    QwenMoeRoutingCapture,
    QwenRoutingBridge,
)
from predictive_cache.scheduler import ExpertScheduler


WARMUP_STEPS = 32
LONGRUN_STEPS = 100


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
    step: int,
) -> list[int]:
    before = len(capture.bridge.collector.events())

    with torch.no_grad():
        capture.capture_forward(
            model,
            input_ids=torch.tensor(
                [[10 + (step % 118)]],
                dtype=torch.long,
            ),
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


def _wait_until_idle(
    pipeline: PrefetchPipeline,
    timeout: float = 10.0,
) -> None:
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if pipeline.engine.unfinished_tasks == 0:
            return

        time.sleep(0.001)

    raise AssertionError(
        "prefetch engine did not become idle "
        f"within {timeout:.1f}s; "
        f"unfinished={pipeline.engine.unfinished_tasks}"
    )


def _warmup_until_prediction(
    capture: QwenMoeRoutingCapture,
    model: Qwen2MoeForCausalLM,
    scheduler: ExpertScheduler,
) -> tuple[list[int], list]:
    current: list[int] = []
    predictions = []

    for step in range(WARMUP_STEPS):
        current = _capture_current(
            capture,
            model,
            step,
        )

        assert current

        predictions = scheduler.plan_prefetch_predictions(
            current
        )

        if predictions:
            return current, predictions

    raise AssertionError(
        "real Qwen predictor did not produce "
        "prefetch candidates during warm-up"
    )


def _make_pipeline(
    cache: PredictiveExpertCache,
    loader,
) -> tuple[
    PrefetchPipeline,
    ExpertScheduler,
]:
    engine = PrefetchEngine(
        loader=loader,
        num_workers=1,
    )

    scheduler = ExpertScheduler(cache)

    pipeline = PrefetchPipeline(
        scheduler,
        engine,
        auto_start=True,
    )

    return pipeline, scheduler


def test_qwen_real_prefetch_state_transition() -> None:
    cache = PredictiveExpertCache(
        CacheConfig(
            capacity=8,
            prediction_top_k=3,
        )
    )

    bridge = QwenRoutingBridge(cache)
    capture = QwenMoeRoutingCapture(bridge)
    model = _build_real_qwen()

    loaded: set[int] = set()

    def loader(task: PrefetchTask) -> None:
        cache.insert(
            task.expert_id,
            size_bytes=1024,
            location="memory",
        )
        loaded.add(task.expert_id)

    pipeline, scheduler = _make_pipeline(
        cache,
        loader,
    )

    try:
        current, predictions = _warmup_until_prediction(
            capture,
            model,
            scheduler,
        )

        predicted_expert = predictions[0].expert_id

        assert cache.get(predicted_expert) is None

        cold_hit = (
            cache.get(predicted_expert) is not None
        )

        assert cold_hit is False

        result = pipeline.process(current)

        matching_tasks = [
            task
            for task in result.submitted_tasks
            if task.expert_id == predicted_expert
        ]

        assert matching_tasks

        _wait_until_idle(pipeline)

        assert predicted_expert in loaded
        assert cache.get(predicted_expert) is not None

        warm_hit = (
            cache.get(predicted_expert) is not None
        )

        assert warm_hit is True

        print()
        print("7-M-1 Real Qwen -> Prefetch State Transition")
        print("-----------------------------------------------")
        print(f"prediction current:    {len(current)}")
        print(f"predicted candidates:   {len(predictions)}")
        print(f"selected expert:        {predicted_expert}")
        print(
            f"submitted prefetches:   "
            f"{len(result.submitted_tasks)}"
        )
        print(f"cold demand hit:        {cold_hit}")
        print(f"warm demand hit:        {warm_hit}")

    finally:
        pipeline.stop()


def test_qwen_real_prefetch_hit_vs_cold_demand() -> None:
    cache = PredictiveExpertCache(
        CacheConfig(
            capacity=8,
            prediction_top_k=3,
        )
    )

    bridge = QwenRoutingBridge(cache)
    capture = QwenMoeRoutingCapture(bridge)
    model = _build_real_qwen()

    def loader(task: PrefetchTask) -> None:
        cache.insert(
            task.expert_id,
            size_bytes=1024,
            location="memory",
        )

    pipeline, scheduler = _make_pipeline(
        cache,
        loader,
    )

    try:
        current, predictions = _warmup_until_prediction(
            capture,
            model,
            scheduler,
        )

        predicted_expert = predictions[0].expert_id

        assert cache.get(predicted_expert) is None

        cold_start = time.perf_counter()

        cold_value = cache.get(predicted_expert)

        cold_elapsed = (
            time.perf_counter() - cold_start
        )

        assert cold_value is None

        result = pipeline.process(current)

        matching_tasks = [
            task
            for task in result.submitted_tasks
            if task.expert_id == predicted_expert
        ]

        assert matching_tasks

        _wait_until_idle(pipeline)

        assert cache.get(predicted_expert) is not None

        warm_start = time.perf_counter()

        warm_value = cache.get(predicted_expert)

        warm_elapsed = (
            time.perf_counter() - warm_start
        )

        assert warm_value is not None

        print()
        print("7-M-2 Real Qwen -> Prefetch Hit vs Cold Demand")
        print("------------------------------------------------")
        print(f"prediction current:    {len(current)}")
        print(f"predicted candidates:   {len(predictions)}")
        print(f"selected expert:        {predicted_expert}")
        print(
            f"submitted prefetches:   "
            f"{len(result.submitted_tasks)}"
        )
        print(f"cold demand:            MISS")
        print(
            f"cold lookup latency:    "
            f"{cold_elapsed * 1000:.3f} ms"
        )
        print(f"prefetched demand:      HIT")
        print(
            f"warm lookup latency:    "
            f"{warm_elapsed * 1000:.3f} ms"
        )

        assert cold_elapsed >= 0.0
        assert warm_elapsed >= 0.0

    finally:
        pipeline.stop()


def test_qwen_real_repeated_demand_benefit() -> None:
    cache = PredictiveExpertCache(
        CacheConfig(
            capacity=8,
            prediction_top_k=3,
        )
    )

    bridge = QwenRoutingBridge(cache)
    capture = QwenMoeRoutingCapture(bridge)
    model = _build_real_qwen()

    def loader(task: PrefetchTask) -> None:
        cache.insert(
            task.expert_id,
            size_bytes=1024,
            location="memory",
        )

    pipeline, scheduler = _make_pipeline(
        cache,
        loader,
    )

    try:
        current, predictions = _warmup_until_prediction(
            capture,
            model,
            scheduler,
        )

        predicted_expert = predictions[0].expert_id

        assert cache.get(predicted_expert) is None

        cold_misses = 0
        warm_hits = 0

        for _ in range(5):
            if cache.get(predicted_expert) is None:
                cold_misses += 1

        assert cold_misses == 5

        result = pipeline.process(current)

        matching_tasks = [
            task
            for task in result.submitted_tasks
            if task.expert_id == predicted_expert
        ]

        assert matching_tasks

        _wait_until_idle(pipeline)

        for _ in range(5):
            if cache.get(predicted_expert) is not None:
                warm_hits += 1

        assert warm_hits == 5

        print()
        print("7-M-3 Real Qwen -> Repeated Demand Benefit")
        print("--------------------------------------------")
        print(f"prediction current:    {len(current)}")
        print(f"predicted candidates:   {len(predictions)}")
        print(f"selected expert:        {predicted_expert}")
        print(f"cold demand misses:     {cold_misses}")
        print(f"prefetched demand hits: {warm_hits}")

    finally:
        pipeline.stop()


def test_qwen_real_longrun_prefetch_benefit_stability() -> None:
    cache = PredictiveExpertCache(
        CacheConfig(
            capacity=8,
            prediction_top_k=3,
        )
    )

    bridge = QwenRoutingBridge(cache)
    capture = QwenMoeRoutingCapture(bridge)
    model = _build_real_qwen()

    total_predictions = 0
    total_submitted = 0
    total_hits = 0
    total_misses = 0
    steps_with_prediction = 0

    def loader(task: PrefetchTask) -> None:
        cache.insert(
            task.expert_id,
            size_bytes=1024,
            location="memory",
        )

    pipeline, scheduler = _make_pipeline(
        cache,
        loader,
    )

    try:
        # Establish predictor history.
        for step in range(WARMUP_STEPS):
            current = _capture_current(
                capture,
                model,
                step,
            )

            assert current

            scheduler.plan_prefetch_predictions(
                current
            )

        for offset in range(LONGRUN_STEPS):
            step = WARMUP_STEPS + offset

            current = _capture_current(
                capture,
                model,
                step,
            )

            assert current

            predictions = (
                scheduler.plan_prefetch_predictions(
                    current
                )
            )

            if not predictions:
                assert pipeline.engine.unfinished_tasks == 0
                continue

            steps_with_prediction += 1
            total_predictions += len(predictions)

            result = pipeline.process(current)

            total_submitted += len(
                result.submitted_tasks
            )

            _wait_until_idle(pipeline)

            predicted_ids = {
                prediction.expert_id
                for prediction in predictions
            }

            for expert_id in predicted_ids:
                if cache.get(expert_id) is not None:
                    total_hits += 1
                else:
                    total_misses += 1

        assert steps_with_prediction > 0
        assert total_predictions > 0
        assert total_submitted > 0

        assert (
            total_hits + total_misses
            == total_predictions
        )

        assert pipeline.engine.unfinished_tasks == 0

        print()
        print(
            "7-M-4 Real Qwen -> Long-Run "
            "Prefetch Benefit Stability"
        )
        print(
            "------------------------------------------"
        )
        print(
            f"warm-up steps:          "
            f"{WARMUP_STEPS}"
        )
        print(
            f"long-run routing steps: "
            f"{LONGRUN_STEPS}"
        )
        print(
            f"steps with prediction:   "
            f"{steps_with_prediction}"
        )
        print(
            f"predicted candidates:    "
            f"{total_predictions}"
        )
        print(
            f"submitted prefetches:   "
            f"{total_submitted}"
        )
        print(
            f"predicted resident hits: "
            f"{total_hits}"
        )
        print(
            f"predicted misses:        "
            f"{total_misses}"
        )
        print(
            f"unfinished tasks:        "
            f"{pipeline.engine.unfinished_tasks}"
        )

    finally:
        pipeline.stop()
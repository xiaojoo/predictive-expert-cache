from __future__ import annotations

import time

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
) -> None:
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if pipeline.engine.unfinished_tasks == 0:
            return

        time.sleep(0.01)

    raise AssertionError(
        "prefetch tasks did not reach terminal state "
        f"within {timeout:.1f}s; "
        f"expected={expected}, "
        f"unfinished={pipeline.engine.unfinished_tasks}"
    )


def test_qwen_real_failure_accounting() -> None:
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
        raise RuntimeError(
            f"intentional failure for expert {task.expert_id}"
        )

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

    try:
        current: list[int] = []
        scheduled_preview = []

        # Warm up the real Qwen routing predictor.
        #
        # The first routing event may legitimately produce no
        # prefetch candidates because the predictor has no history
        # yet. Keep observing real routing until we have a valid
        # predictive-pre-fetch opportunity.
        for step in range(32):
            current = _capture_current(
                capture,
                model,
                torch.tensor(
                    [[10 + (step % 118)]],
                    dtype=torch.long,
                ),
                step=step,
            )

            assert current

            scheduled_preview = (
                scheduler.plan_prefetch_predictions(
                    current
                )
            )

            if scheduled_preview:
                break

        assert scheduled_preview, (
            "real Qwen routing did not produce any "
            "prefetch candidates after predictor warm-up"
        )

        result = pipeline.process(current)

        submitted = len(result.submitted_tasks)

        assert submitted > 0

        _wait_until_terminal(
            pipeline,
            expected=submitted,
        )

        completed = 0
        failed = 0
        cancelled = 0

        for task in result.submitted_tasks:
            status = pipeline.engine.task_status(task.expert_id)

            if status == PrefetchStatus.COMPLETED:
                completed += 1
            elif status == PrefetchStatus.FAILED:
                failed += 1
            elif status == PrefetchStatus.CANCELLED:
                cancelled += 1

        terminal_accounted = (
            completed
            + failed
            + cancelled
        )

        resident = sum(
            1
            for task in result.submitted_tasks
            if cache.get(task.expert_id) is not None
        )

        assert failed == submitted
        assert completed == 0
        assert cancelled == 0

        assert terminal_accounted == submitted

        assert pipeline.engine.unfinished_tasks == 0

        assert resident == completed
        assert resident == 0

        print()
        print("7-L-4 Real Qwen -> Failure Accounting")
        print("--------------------------------------")
        print(f"routed current experts: {len(current)}")
        print(
            f"scheduled requests:     "
            f"{len(result.scheduled_requests)}"
        )
        print(f"submitted tasks:        {submitted}")
        print(f"completed:              {completed}")
        print(f"failed:                 {failed}")
        print(f"cancelled:              {cancelled}")
        print(
            f"terminal accounted:     "
            f"{terminal_accounted}"
        )
        print(
            f"unfinished tasks:       "
            f"{pipeline.engine.unfinished_tasks}"
        )
        print(f"resident:               {resident}")

    finally:
        pipeline.stop()
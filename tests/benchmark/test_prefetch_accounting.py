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


def _wait_terminal(
    engine: PrefetchEngine,
    expert_ids: set[int],
    timeout: float = 5.0,
) -> None:
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if all(
            engine.task_status(expert_id)
            in {
                PrefetchStatus.COMPLETED,
                PrefetchStatus.FAILED,
                PrefetchStatus.CANCELLED,
            }
            for expert_id in expert_ids
        ):
            return

        time.sleep(0.01)

    raise AssertionError(
        "prefetch tasks did not reach terminal state: "
        f"{sorted(expert_ids)}"
    )


def test_real_qwen_prefetch_accounting_invariants() -> None:
    """
    7-L-1:

        Real Qwen routing
            -> scheduler
            -> pipeline
            -> engine
            -> terminal lifecycle
            -> accounting invariants

    This test intentionally does not add a new metrics subsystem.

    It verifies that the existing runtime state is internally
    consistent and that terminal task accounting is lossless.
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

    executions: dict[int, int] = {}

    def loader(task: PrefetchTask) -> None:
        expert_id = int(task.expert_id)

        executions[expert_id] = (
            executions.get(expert_id, 0) + 1
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
    ]

    routed_currents: list[list[int]] = []

    try:
        for step, input_ids in enumerate(inputs):
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

            cache.observe(current)
            routed_currents.append(current)

        assert routed_currents

        selected_current: list[int] | None = None

        for candidate_current in routed_currents:
            requests = scheduler.plan_prefetch_predictions(
                candidate_current
            )

            if requests:
                selected_current = candidate_current
                break

        assert selected_current is not None

        result = pipeline.process(selected_current)

        assert result.scheduled_requests
        assert result.submitted_tasks

        submitted_ids = {
            int(task.expert_id)
            for task in result.submitted_tasks
        }

        assert submitted_ids

        _wait_terminal(
            engine,
            submitted_ids,
        )

        completed = sum(
            engine.task_status(expert_id)
            == PrefetchStatus.COMPLETED
            for expert_id in submitted_ids
        )

        failed = sum(
            engine.task_status(expert_id)
            == PrefetchStatus.FAILED
            for expert_id in submitted_ids
        )

        cancelled = sum(
            engine.task_status(expert_id)
            == PrefetchStatus.CANCELLED
            for expert_id in submitted_ids
        )

        terminal = completed + failed + cancelled

        assert terminal == len(submitted_ids)

        assert engine.unfinished_tasks == 0

        # Every accepted task must have exactly one terminal outcome.
        assert (
            completed
            + failed
            + cancelled
            == len(result.submitted_tasks)
        )

        # This benchmark's loader is deterministic and successful,
        # therefore every submitted task must complete.
        assert failed == 0
        assert cancelled == 0
        assert completed == len(submitted_ids)

        # Completion is a lifecycle property, not "usefulness".
        # We only assert residency here because this loader inserts
        # every completed expert into RAM.
        assert all(
            cache.contains(expert_id)
            for expert_id in submitted_ids
        )

        assert all(
            executions.get(expert_id) == 1
            for expert_id in submitted_ids
        )

        print()
        print("7-L-1 Real Qwen -> Prefetch Accounting")
        print("---------------------------------------")
        print(
            f"routed current experts: {len(selected_current)}"
        )
        print(
            f"scheduled requests:     "
            f"{len(result.scheduled_requests)}"
        )
        print(
            f"submitted tasks:        "
            f"{len(result.submitted_tasks)}"
        )
        print(f"completed:              {completed}")
        print(f"failed:                 {failed}")
        print(f"cancelled:              {cancelled}")
        print(
            f"terminal accounted:     {terminal}"
        )
        print(
            f"unfinished tasks:       "
            f"{engine.unfinished_tasks}"
        )
        print(
            f"resident:               "
            f"{sum(cache.contains(i) for i in submitted_ids)}"
        )

    finally:
        if engine.running:
            engine.stop(wait=True)
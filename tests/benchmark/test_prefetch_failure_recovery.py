from __future__ import annotations

import threading
import time

import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from predictive_cache.cache import PredictiveExpertCache
from predictive_cache.prefetch.admission import (
    AdmissionController,
    AdmissionReason,
)
from predictive_cache.prefetch.engine import PrefetchEngine
from predictive_cache.prefetch.pipeline import PrefetchPipeline
from predictive_cache.prefetch.types import (
    PrefetchSource,
    PrefetchStatus,
    PrefetchTask,
    PrefetchTarget,
)
from predictive_cache.routing import (
    QwenMoeRoutingCapture,
    QwenRoutingBridge,
)
from predictive_cache.scheduler import ExpertScheduler
from predictive_cache.types import CacheConfig


def _wait_for_status(
    engine: PrefetchEngine,
    expert_id: int,
    expected: PrefetchStatus,
    *,
    timeout: float = 1.0,
) -> None:
    deadline = time.perf_counter() + timeout

    while time.perf_counter() < deadline:
        if engine.task_status(expert_id) == expected:
            return

        time.sleep(0.00005)

    assert engine.task_status(expert_id) == expected


def _make_task(expert_id: int) -> PrefetchTask:
    return PrefetchTask(
        expert_id=expert_id,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
        priority=1.0,
        confidence=1.0,
        estimated_distance=1,
    )


def test_prefetch_failure_reaches_failed_state() -> None:
    """
    7-K-1:

        PrefetchEngine
            -> loader failure
            -> FAILED

    The failure must be represented as a terminal task state and
    must not leave the task queued/running or leave unfinished queue
    work behind.
    """

    expert_id = 7001

    def loader(task: PrefetchTask) -> None:
        raise RuntimeError("synthetic prefetch failure")

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )

    task = _make_task(expert_id)

    try:
        engine.start()

        assert engine.submit(task) is True

        _wait_for_status(
            engine,
            expert_id,
            PrefetchStatus.FAILED,
        )

        assert engine.task_status(expert_id) == PrefetchStatus.FAILED
        assert expert_id in engine.failed_tasks
        assert expert_id not in engine.queued_tasks
        assert expert_id not in engine.running_tasks
        assert engine.unfinished_tasks == 0

    finally:
        if engine.running:
            engine.stop(wait=True)


def test_failed_prefetch_is_admissible_for_retry() -> None:
    """
    7-K-2:

        submit
            -> FAILED
            -> AdmissionController.evaluate(same task)
            -> ACCEPT

    FAILED is terminal, but it must not behave like an active
    duplicate QUEUED/RUNNING task.
    """

    expert_id = 7002

    def loader(task: PrefetchTask) -> None:
        raise RuntimeError("synthetic prefetch failure")

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )

    admission = AdmissionController()
    task = _make_task(expert_id)

    try:
        engine.start()

        assert engine.submit(task) is True

        _wait_for_status(
            engine,
            expert_id,
            PrefetchStatus.FAILED,
        )

        decision = admission.evaluate(task, engine)

        assert decision.admitted is True
        assert decision.accepted is True
        assert decision.reason == AdmissionReason.ACCEPT

    finally:
        if engine.running:
            engine.stop(wait=True)


def test_failed_prefetch_can_retry_and_complete() -> None:
    """
    7-K-3:

        attempt #1
            -> FAILED

        retry
            -> COMPLETED

    This locks the failure-recovery lifecycle without modifying
    production Engine semantics.
    """

    expert_id = 7003
    attempts = 0

    def loader(task: PrefetchTask) -> None:
        nonlocal attempts

        attempts += 1

        if attempts == 1:
            raise RuntimeError("synthetic transient failure")

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )

    admission = AdmissionController()
    task = _make_task(expert_id)

    try:
        engine.start()

        # Attempt #1 must fail.
        assert engine.submit(task) is True

        _wait_for_status(
            engine,
            expert_id,
            PrefetchStatus.FAILED,
        )

        assert engine.task_status(expert_id) == PrefetchStatus.FAILED
        assert attempts == 1
        assert engine.unfinished_tasks == 0

        # FAILED must be retryable.
        decision = admission.evaluate(task, engine)

        assert decision.admitted is True
        assert decision.reason == AdmissionReason.ACCEPT

        # Attempt #2 must complete.
        assert engine.submit(task) is True

        _wait_for_status(
            engine,
            expert_id,
            PrefetchStatus.COMPLETED,
        )

        assert engine.task_status(expert_id) == PrefetchStatus.COMPLETED
        assert attempts == 2
        assert expert_id in engine.completed_tasks
        assert expert_id not in engine.running_tasks
        assert expert_id not in engine.queued_tasks
        assert engine.unfinished_tasks == 0

    finally:
        if engine.running:
            engine.stop(wait=True)


def test_queued_prefetch_can_be_cancelled() -> None:
    """
    7-K-4:

        QUEUED -> CANCELLED

    A blocker occupies the only worker so the target task remains
    queued long enough for cancellation to be deterministic.
    """

    blocker_id = 7004
    target_id = 7005

    blocker_started = threading.Event()
    blocker_release = threading.Event()

    executed: list[int] = []

    def loader(task: PrefetchTask) -> None:
        executed.append(task.expert_id)

        if task.expert_id == blocker_id:
            blocker_started.set()
            assert blocker_release.wait(timeout=1.0)

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )

    blocker = _make_task(blocker_id)
    target = _make_task(target_id)

    try:
        engine.start()

        assert engine.submit(blocker) is True

        assert blocker_started.wait(timeout=1.0)

        assert engine.submit(target) is True

        _wait_for_status(
            engine,
            target_id,
            PrefetchStatus.QUEUED,
        )

        assert target_id in engine.queued_tasks
        assert target_id not in executed

        assert engine.cancel(target_id) is True

        assert engine.task_status(target_id) == PrefetchStatus.CANCELLED
        assert target_id in engine.cancelled_tasks
        assert target_id not in engine.queued_tasks
        assert target_id not in engine.running_tasks
        assert target_id not in executed

        # Cancellation itself must retire the queue's unfinished
        # accounting for the cancelled task.
        assert engine.unfinished_tasks == 1

    finally:
        blocker_release.set()

        if engine.running:
            engine.stop(wait=True)

    assert executed == [blocker_id]


def test_running_prefetch_cannot_be_cancelled() -> None:
    """
    7-K-4:

        RUNNING -> cancel() == False
        RUNNING remains RUNNING.

    PrefetchEngine deliberately does not forcefully terminate a
    worker thread already executing the loader.
    """

    expert_id = 7006

    started = threading.Event()
    release = threading.Event()

    def loader(task: PrefetchTask) -> None:
        started.set()
        assert release.wait(timeout=1.0)

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )

    task = _make_task(expert_id)

    try:
        engine.start()

        assert engine.submit(task) is True

        assert started.wait(timeout=1.0)

        _wait_for_status(
            engine,
            expert_id,
            PrefetchStatus.RUNNING,
        )

        assert engine.cancel(expert_id) is False
        assert engine.task_status(expert_id) == PrefetchStatus.RUNNING
        assert expert_id in engine.running_tasks

    finally:
        release.set()

        if engine.running:
            engine.stop(wait=True)

    assert engine.task_status(expert_id) == PrefetchStatus.COMPLETED


def test_real_qwen_prefetch_failure_recovery() -> None:
    """
    7-K-5:

        Real Qwen MoE routing
            -> ExpertScheduler
            -> PrefetchPipeline
            -> PrefetchEngine
            -> first attempt FAILED
            -> same routing state retry
            -> COMPLETED
            -> RAM resident

    The routing state is produced by an actual Qwen3-MoE forward pass.

    The loader simulates a transient failure:
        attempt #1 -> FAILED
        attempt #2 -> COMPLETED

    No production code is modified.
    """

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

    cache = PredictiveExpertCache(
        CacheConfig(
            capacity=8,
            prediction_top_k=3,
        )
    )

    # Exact construction already validated by 7-J.
    bridge = QwenRoutingBridge(cache)
    capture = QwenMoeRoutingCapture(bridge)
    scheduler = ExpertScheduler(cache)

    engine_attempts: dict[int, int] = {}

    def loader(task: PrefetchTask) -> None:
        expert_id = int(task.expert_id)

        engine_attempts[expert_id] = (
            engine_attempts.get(expert_id, 0) + 1
        )

        # Every expert fails exactly once.
        if engine_attempts[expert_id] == 1:
            raise RuntimeError(
                f"synthetic transient failure for expert {expert_id}"
            )

        # Retry succeeds and makes the expert resident.
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
        # ---------------------------------------------------------
        # Establish REAL Qwen routing history.
        #
        # Do not use bridge.capture():
        # QwenRoutingBridge has no such API.
        #
        # capture_forward() executes the actual Qwen model and
        # records routing events in bridge.collector.
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # Real Qwen routing is input-dependent.
        #
        # Select the first real routing state for which the
        # scheduler has an actual prefetch prediction.
        # ---------------------------------------------------------
        selected_current: list[int] | None = None

        for candidate_current in routed_currents:
            requests = scheduler.plan_prefetch_predictions(
                candidate_current
            )

            if requests:
                selected_current = candidate_current
                break

        assert selected_current is not None

        # ---------------------------------------------------------
        # First real-Qwen pipeline invocation.
        #
        # Every submitted task will fail exactly once.
        # ---------------------------------------------------------
        first = pipeline.process(selected_current)

        assert first.scheduled_requests
        assert first.submitted_tasks

        first_ids = {
            task.expert_id
            for task in first.submitted_tasks
        }

        assert first_ids

        # ---------------------------------------------------------
        # First attempt -> FAILED.
        # ---------------------------------------------------------
        for expert_id in first_ids:
            _wait_for_status(
                engine,
                expert_id,
                PrefetchStatus.FAILED,
                timeout=5.0,
            )

        assert all(
            engine.task_status(expert_id) == PrefetchStatus.FAILED
            for expert_id in first_ids
        )

        assert all(
            engine_attempts.get(expert_id) == 1
            for expert_id in first_ids
        )

        # Failed prefetches must not become resident.
        assert all(
            not cache.contains(expert_id)
            for expert_id in first_ids
        )

        # ---------------------------------------------------------
        # Retry the EXACT SAME real-Qwen routing state.
        #
        # Cache state for these experts is unchanged because every
        # first attempt failed. AdmissionController must therefore
        # allow the FAILED tasks to be submitted again.
        # ---------------------------------------------------------
        second = pipeline.process(selected_current)

        assert second.scheduled_requests
        assert second.submitted_tasks

        second_ids = {
            task.expert_id
            for task in second.submitted_tasks
        }

        assert second_ids
        assert second_ids == first_ids

        # ---------------------------------------------------------
        # Retry -> COMPLETED.
        # ---------------------------------------------------------
        for expert_id in second_ids:
            _wait_for_status(
                engine,
                expert_id,
                PrefetchStatus.COMPLETED,
                timeout=5.0,
            )

        assert all(
            engine.task_status(expert_id)
            == PrefetchStatus.COMPLETED
            for expert_id in second_ids
        )

        assert all(
            engine_attempts.get(expert_id) == 2
            for expert_id in second_ids
        )

        # ---------------------------------------------------------
        # Recovery must make every expert resident.
        # ---------------------------------------------------------
        assert all(
            cache.contains(expert_id)
            for expert_id in second_ids
        )

        assert engine.unfinished_tasks == 0

        print()
        print("7-K-5 Real Qwen -> Failure / Recovery")
        print("---------------------------------------")
        print(
            f"routed current experts: {len(selected_current)}"
        )
        print(f"first scheduled:        {len(first.scheduled_requests)}")
        print(f"first submitted:        {len(first.submitted_tasks)}")
        print(f"first failed:           {len(first_ids)}")
        print(f"second scheduled:       {len(second.scheduled_requests)}")
        print(f"second submitted:       {len(second.submitted_tasks)}")
        print(f"second completed:       {len(second_ids)}")
        print(f"resident after retry:   {len(second_ids)}")
        print(f"attempts:               {engine_attempts}")

    finally:
        if engine.running:
            engine.stop(wait=True)
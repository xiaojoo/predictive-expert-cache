from __future__ import annotations

import threading
import time

from predictive_cache.prefetch import (
    PrefetchEngine,
    PrefetchSource,
    PrefetchStatus,
    PrefetchTarget,
    PrefetchTask,
)
from predictive_cache.prefetch.admission import (
    AdmissionController,
    AdmissionReason,
)
from predictive_cache.prefetch.admission_stats import AdmissionStats


def _make_task(
    expert_id: int,
    *,
    priority: float = 1.0,
    confidence: float = 0.9,
) -> PrefetchTask:
    return PrefetchTask(
        expert_id=expert_id,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
        priority=priority,
        confidence=confidence,
    )


def _wait_for_status(
    engine: PrefetchEngine,
    expert_id: int,
    status: PrefetchStatus,
    timeout: float = 2.0,
) -> bool:
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if engine.task_status(expert_id) == status:
            return True

        time.sleep(0.001)

    return False


def test_admission_duplicate_suppression_1000_cycles() -> None:
    started = threading.Event()
    release = threading.Event()

    def loader(task: PrefetchTask) -> None:
        started.set()
        release.wait(timeout=2.0)

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )

    admission = AdmissionController()
    stats = AdmissionStats()

    engine.start()

    try:
        for cycle in range(1000):
            expert_id = cycle

            task = _make_task(
                expert_id,
                priority=1.0,
                confidence=0.9,
            )

            # -------------------------------------------------
            # First admission must succeed.
            # -------------------------------------------------

            decision = admission.evaluate(
                task,
                engine,
            )

            assert decision.admitted is True
            assert decision.reason == AdmissionReason.ACCEPT

            stats.record(decision)

            assert engine.submit(task) is True

            # -------------------------------------------------
            # Wait until task is actually RUNNING.
            # -------------------------------------------------

            assert started.wait(timeout=2.0)

            assert engine.task_status(
                expert_id
            ) == PrefetchStatus.RUNNING

            # -------------------------------------------------
            # Same expert while RUNNING must be rejected.
            # -------------------------------------------------

            duplicate = _make_task(
                expert_id,
                priority=10.0,
                confidence=1.0,
            )

            duplicate_decision = admission.evaluate(
                duplicate,
                engine,
            )

            assert duplicate_decision.admitted is False

            assert (
                duplicate_decision.reason
                == AdmissionReason.REJECT_DUPLICATE_RUNNING
            )

            stats.record(duplicate_decision)

            # -------------------------------------------------
            # Release current task.
            # -------------------------------------------------

            release.set()

            assert _wait_for_status(
                engine,
                expert_id,
                PrefetchStatus.COMPLETED,
            )

            # Reset events for next cycle.
            started.clear()
            release.clear()

            # -------------------------------------------------
            # COMPLETED expert must be admissible again.
            # -------------------------------------------------

            retry = _make_task(
                expert_id,
                priority=1.0,
                confidence=0.9,
            )

            retry_decision = admission.evaluate(
                retry,
                engine,
            )

            assert retry_decision.admitted is True
            assert retry_decision.reason == (
                AdmissionReason.ACCEPT
            )

            stats.record(retry_decision)

            # Do not submit the retry here.
            #
            # This cycle intentionally verifies that a completed
            # task does not remain permanently suppressed.
            #
            # The next cycle uses a different expert id.

        assert stats.total == 3000
        assert stats.accepted == 2000
        assert stats.rejected == 1000
        assert stats.reject_duplicate_running == 1000

        assert stats.reject_duplicate_queued == 0
        assert stats.reject_low_priority == 0
        assert stats.reject_low_confidence == 0
        assert stats.reject_queue_full == 0

        assert stats.acceptance_rate == 2 / 3
        assert stats.rejection_rate == 1 / 3

    finally:
        release.set()
        engine.stop()


def test_admission_state_does_not_grow_from_duplicate_checks() -> None:
    started = threading.Event()
    release = threading.Event()

    def loader(task: PrefetchTask) -> None:
        started.set()
        release.wait(timeout=2.0)

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )

    admission = AdmissionController()

    engine.start()

    try:
        task = _make_task(1)

        assert admission.admit(
            task,
            engine,
        ) is True

        assert engine.submit(task) is True

        assert started.wait(timeout=2.0)

        assert engine.task_status(1) == (
            PrefetchStatus.RUNNING
        )

        initial_states = engine.task_states()

        assert set(initial_states) == {1}

        # Perform many duplicate checks.
        for _ in range(1000):
            duplicate = _make_task(
                1,
                priority=10.0,
                confidence=1.0,
            )

            decision = admission.evaluate(
                duplicate,
                engine,
            )

            assert decision.admitted is False
            assert decision.reason == (
                AdmissionReason.REJECT_DUPLICATE_RUNNING
            )

            # Admission itself must not mutate engine state.

            assert engine.task_states() == initial_states

        assert len(engine.task_states()) == 1
        assert engine.queued_tasks == []
        assert engine.running_tasks == [1]

    finally:
        release.set()
        engine.stop()


def test_admission_duplicate_suppression_is_deterministic() -> None:
    started = threading.Event()
    release = threading.Event()

    def loader(task: PrefetchTask) -> None:
        started.set()
        release.wait(timeout=2.0)

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )

    admission = AdmissionController()

    engine.start()

    try:
        task = _make_task(42)

        assert engine.submit(task) is True

        assert started.wait(timeout=2.0)

        duplicate = _make_task(
            42,
            priority=10.0,
            confidence=1.0,
        )

        decisions = [
            admission.evaluate(
                duplicate,
                engine,
            )
            for _ in range(1000)
        ]

        assert all(
            decision.admitted is False
            for decision in decisions
        )

        assert all(
            decision.reason
            == AdmissionReason.REJECT_DUPLICATE_RUNNING
            for decision in decisions
        )

        assert len(set(decisions)) == 1

    finally:
        release.set()
        engine.stop()
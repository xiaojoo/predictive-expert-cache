from dataclasses import dataclass
from enum import Enum

from .engine import PrefetchEngine
from .types import PrefetchTask


class AdmissionReason(str, Enum):
    ACCEPT = "accept"
    REJECT_LOW_PRIORITY = "reject_low_priority"
    REJECT_LOW_CONFIDENCE = "reject_low_confidence"
    REJECT_DUPLICATE_QUEUED = "reject_duplicate_queued"
    REJECT_DUPLICATE_RUNNING = "reject_duplicate_running"
    REJECT_QUEUE_FULL = "reject_queue_full"


@dataclass(frozen=True)
class AdmissionDecision:
    admitted: bool
    reason: AdmissionReason

    @property
    def accepted(self) -> bool:
        return self.admitted


class AdmissionController:
    def __init__(
        self,
        *,
        min_priority: float = 0.0,
        min_confidence: float = 0.0,
    ):
        if min_priority < 0:
            raise ValueError("min_priority must be >= 0")

        if min_confidence < 0:
            raise ValueError("min_confidence must be >= 0")

        if min_confidence > 1:
            raise ValueError("min_confidence must be <= 1")

        self.min_priority = min_priority
        self.min_confidence = min_confidence

    def evaluate(
        self,
        task: PrefetchTask,
        engine: PrefetchEngine,
    ) -> AdmissionDecision:
        """
        Evaluate whether a task should be admitted.

        The evaluation order is intentionally kept stable:

        1. priority
        2. confidence
        3. queued duplicate
        4. running duplicate
        5. queue full
        6. accept
        """

        if task.priority < self.min_priority:
            return AdmissionDecision(
                admitted=False,
                reason=AdmissionReason.REJECT_LOW_PRIORITY,
            )

        if task.confidence < self.min_confidence:
            return AdmissionDecision(
                admitted=False,
                reason=AdmissionReason.REJECT_LOW_CONFIDENCE,
            )

        if task.expert_id in engine.queued_tasks:
            return AdmissionDecision(
                admitted=False,
                reason=AdmissionReason.REJECT_DUPLICATE_QUEUED,
            )

        if task.expert_id in engine.running_tasks:
            return AdmissionDecision(
                admitted=False,
                reason=AdmissionReason.REJECT_DUPLICATE_RUNNING,
            )

        if engine.queue_full:
            return AdmissionDecision(
                admitted=False,
                reason=AdmissionReason.REJECT_QUEUE_FULL,
            )

        return AdmissionDecision(
            admitted=True,
            reason=AdmissionReason.ACCEPT,
        )

    def admit(
        self,
        task: PrefetchTask,
        engine: PrefetchEngine,
    ) -> bool:
        """
        Backward-compatible boolean admission API.
        """

        return self.evaluate(task, engine).admitted
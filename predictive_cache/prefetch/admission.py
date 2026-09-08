from __future__ import annotations

from .engine import PrefetchEngine
from .types import PrefetchTask


class AdmissionController:
    """
    Prefetch task admission controller.

    4D-7-1:

    Admission decisions are based on:

    - priority threshold
    - confidence threshold
    - duplicate active task
    - queue capacity

    The controller only decides whether a task should be admitted.
    It does not submit the task to the engine.

    Priority eviction is intentionally NOT handled here.
    """

    def __init__(
        self,
        *,
        min_priority: float = 0.0,
        min_confidence: float = 0.0,
    ) -> None:
        self._validate_threshold(
            min_priority,
            name="min_priority",
        )
        self._validate_threshold(
            min_confidence,
            name="min_confidence",
        )

        if min_confidence > 1.0:
            raise ValueError("min_confidence must be <= 1.0")

        self._min_priority = min_priority
        self._min_confidence = min_confidence

    @property
    def min_priority(self) -> float:
        return self._min_priority

    @property
    def min_confidence(self) -> float:
        return self._min_confidence

    def admit(
        self,
        task: PrefetchTask,
        engine: PrefetchEngine,
    ) -> bool:
        """
        Decide whether a PrefetchTask should enter the prefetch queue.

        Returns:
            True:
                Task passes all admission checks.

            False:
                Task should not be submitted.
        """

        # -----------------------------------------------------
        # 1. Priority threshold
        # -----------------------------------------------------
        if task.priority < self._min_priority:
            return False

        # -----------------------------------------------------
        # 2. Confidence threshold
        # -----------------------------------------------------
        if task.confidence < self._min_confidence:
            return False

        # -----------------------------------------------------
        # 3. Duplicate active task
        #
        # A task is considered duplicate when the same expert
        # is already queued or running.
        #
        # Completed / failed / cancelled tasks are intentionally
        # not considered duplicates.
        # -----------------------------------------------------
        if task.expert_id in engine.queued_tasks:
            return False

        if task.expert_id in engine.running_tasks:
            return False

        # -----------------------------------------------------
        # 4. Queue capacity
        # -----------------------------------------------------
        if engine.queue_full:
            return False

        return True

    @staticmethod
    def _validate_threshold(
        value: float,
        *,
        name: str,
    ) -> None:
        if value < 0.0:
            raise ValueError(f"{name} must be >= 0.0")
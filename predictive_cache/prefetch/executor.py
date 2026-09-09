from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

from .task import PrefetchTask, TaskResult, TaskState


class PrefetchExecutor(ABC):
    """Abstract executor for prefetch tasks."""

    @abstractmethod
    def execute(self, task: PrefetchTask) -> TaskResult:
        """Execute one prefetch task."""
        raise NotImplementedError


class InMemoryPrefetchExecutor(PrefetchExecutor):
    """
    Simple in-memory executor used for testing.

    It records completed experts instead of loading real model data.
    """

    def __init__(
        self,
        loader: Optional[Callable[[PrefetchTask], None]] = None,
    ) -> None:
        self._loaded_experts: set[int] = set()
        self._executed_tasks: list[PrefetchTask] = []
        self._loader = loader

    def execute(self, task: PrefetchTask) -> TaskResult:
        try:
            if self._loader is not None:
                self._loader(task)

            self._loaded_experts.add(task.expert_id)
            self._executed_tasks.append(task)

            return TaskResult(
                task=task,
                state=TaskState.COMPLETED,
            )

        except Exception as exc:
            return TaskResult(
                task=task,
                state=TaskState.FAILED,
                error=str(exc),
            )

    def is_loaded(self, expert_id: int) -> bool:
        return expert_id in self._loaded_experts

    @property
    def loaded_experts(self) -> set[int]:
        return set(self._loaded_experts)

    @property
    def executed_tasks(self) -> list[PrefetchTask]:
        return list(self._executed_tasks)

    def clear(self) -> None:
        self._loaded_experts.clear()
        self._executed_tasks.clear()


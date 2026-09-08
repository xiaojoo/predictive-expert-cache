from __future__ import annotations

from .executor import PrefetchExecutor
from .queue import PrefetchQueue
from .task import TaskResult


class PrefetchWorker:
    """Consume tasks from a prefetch queue and execute them."""

    def __init__(
        self,
        queue: PrefetchQueue,
        executor: PrefetchExecutor,
    ) -> None:
        self._queue = queue
        self._executor = executor
        self._results: list[TaskResult] = []

    def run_once(self) -> TaskResult | None:
        """
        Execute one queued task.

        Returns None when the queue is empty.
        """
        task = self._queue.get(timeout=0)

        if task is None:
            return None

        result = self._executor.execute(task)
        self._results.append(result)

        return result

    def run(self, max_tasks: int | None = None) -> list[TaskResult]:
        """
        Execute queued tasks until the queue is empty.

        If max_tasks is provided, execute at most that many tasks.
        """
        results: list[TaskResult] = []

        while max_tasks is None or len(results) < max_tasks:
            result = self.run_once()

            if result is None:
                break

            results.append(result)

        return results

    @property
    def results(self) -> list[TaskResult]:
        """Return all results produced by this worker."""
        return list(self._results)

    def clear_results(self) -> None:
        """Clear previously recorded worker results."""
        self._results.clear()
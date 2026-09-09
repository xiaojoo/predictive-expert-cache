from __future__ import annotations

from typing import Protocol

from .types import (
    PrefetchSource,
    PrefetchTask,
    PrefetchTarget,
)


class PrefetchTransferHandler(Protocol):
    """Interface for one concrete prefetch transfer operation."""

    def __call__(self, task: PrefetchTask) -> None:
        """Execute one transfer operation."""
        ...


class PrefetchTransferExecutor:
    """
    Execute prefetch tasks according to their source/target pair.

    This is only the routing layer.

    Actual NVMe, RAM and GPU transfer implementations can be
    introduced later without changing PrefetchEngine.
    """

    def __init__(
            self,
            default_handler: PrefetchTransferHandler,
    ) -> None:
        self._default_handler = default_handler
        self._handlers: dict[
            tuple[PrefetchSource, PrefetchTarget],
            PrefetchTransferHandler,
        ] = {}

    def register(
        self,
        source: PrefetchSource,
        target: PrefetchTarget,
        handler: PrefetchTransferHandler,
    ) -> None:
        """Register a handler for one source/target route."""
        self._handlers[(source, target)] = handler

    def execute(
        self,
        task: PrefetchTask,
    ) -> None:
        """
        Execute the handler associated with the task route.

        If no specialized handler has been registered, fall back
        to the default loader.
        """
        handler = self._handlers.get(
            (task.source, task.target),
            self._default_handler,
        )

        handler(task)
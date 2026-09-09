from __future__ import annotations

from typing import Callable

from .types import (
    PrefetchSource,
    PrefetchTask,
    PrefetchTarget,
)


TransferHandler = Callable[[PrefetchTask], None]


class PrefetchTransferExecutor:
    """
    Execute prefetch tasks according to their source/target pair.

    This is only the routing layer.

    Actual NVMe, RAM and GPU transfer implementations can be
    introduced later without changing PrefetchEngine.
    """

    def __init__(
        self,
        default_handler: TransferHandler,
    ) -> None:
        self._default_handler = default_handler
        self._handlers: dict[
            tuple[PrefetchSource, PrefetchTarget],
            TransferHandler,
        ] = {}

    def register(
        self,
        source: PrefetchSource,
        target: PrefetchTarget,
        handler: TransferHandler,
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
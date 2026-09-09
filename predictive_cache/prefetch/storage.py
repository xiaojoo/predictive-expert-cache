from __future__ import annotations

from typing import Protocol


class PrefetchStorage(Protocol):
    """Source storage for prefetch data."""

    def load(self, expert_id: int) -> object:
        """Load one expert from the source storage."""
        ...


class PrefetchRam(Protocol):
    """Target RAM storage for prefetched data."""

    def store(self, expert_id: int, data: object) -> None:
        """Store one expert in RAM."""
        ...


class NvmeToRamHandler:
    """Transfer one expert from storage into RAM."""

    def __init__(
        self,
        storage: PrefetchStorage,
        ram: PrefetchRam,
    ) -> None:
        self._storage = storage
        self._ram = ram

    def __call__(self, task) -> None:
        data = self._storage.load(task.expert_id)
        self._ram.store(task.expert_id, data)
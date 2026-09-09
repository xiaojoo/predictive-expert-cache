from __future__ import annotations

from typing import Protocol

from predictive_cache.storage.expert_record import ExpertRecord
from predictive_cache.storage.expert_store import (
    ExpertLocation,
    ExpertStore,
)

from .types import PrefetchTask


class PrefetchStorage(Protocol):
    """Source storage for prefetch data."""

    def load(self, expert_id: int) -> object:
        """Load one expert payload from the source storage."""
        ...


class PrefetchRam(Protocol):
    """Target RAM storage for prefetched data."""

    def store(self, expert_id: int, data: object) -> None:
        """Store one expert payload in RAM."""
        ...


class ExpertStorePrefetchStorage:
    """
    Adapt an existing ExpertStore as a prefetch source.

    The existing ExpertStore owns the ExpertRecord lifecycle.
    Prefetch only consumes the record payload.
    """

    def __init__(
        self,
        store: ExpertStore,
    ) -> None:
        self._store = store

    def load(self, expert_id: int) -> object:
        record = self._store.get(expert_id)

        if record is None:
            raise KeyError(
                f"expert {expert_id!r} is not present in storage"
            )

        return record.payload


class ExpertStorePrefetchRam:
    """
    Adapt an existing ExpertStore as a prefetch RAM target.
    """

    def __init__(
        self,
        store: ExpertStore,
    ) -> None:
        self._store = store

    def store(
        self,
        expert_id: int,
        data: object,
    ) -> None:
        self._store.put(
            ExpertRecord(
                expert_id=expert_id,
                location=ExpertLocation.RAM,
                payload=data,
            )
        )


class NvmeToRamHandler:
    """Transfer one expert from storage into RAM."""

    def __init__(
        self,
        storage: PrefetchStorage,
        ram: PrefetchRam,
    ) -> None:
        self._storage = storage
        self._ram = ram

    def __call__(self, task: PrefetchTask) -> None:
        data = self._storage.load(task.expert_id)
        self._ram.store(task.expert_id, data)
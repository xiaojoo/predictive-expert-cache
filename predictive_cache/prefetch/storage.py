from __future__ import annotations

from typing import Protocol

from predictive_cache.storage.expert_record import ExpertRecord
from predictive_cache.storage.expert_store import ExpertLocation, ExpertStore

from .transfer import PrefetchTransferExecutor, PrefetchTransferHandler
from .types import PrefetchSource, PrefetchTarget, PrefetchTask


class PrefetchStorage(Protocol):
    def load(self, expert_id: int) -> object: ...


class PrefetchRam(Protocol):
    def store(self, expert_id: int, data: object) -> None: ...


class ExpertStorePrefetchStorage:
    def __init__(self, store: ExpertStore) -> None:
        self._store = store

    def load(self, expert_id: int) -> object:
        record = self._store.get(expert_id)

        if record is None:
            raise KeyError(
                f"expert {expert_id!r} is not present in storage"
            )

        return record.payload


class ExpertStorePrefetchRam:
    def __init__(self, store: ExpertStore) -> None:
        self._store = store

    def store(self, expert_id: int, data: object) -> None:
        self._store.put(
            ExpertRecord(
                expert_id=expert_id,
                location=ExpertLocation.RAM,
                payload=data,
            )
        )


class NvmeToRamHandler:
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


class RamToGpuHandler:
    """
    Transfer an expert from RAM storage to GPU storage.

    The RAM record is left untouched. A separate ExpertRecord is created
    for the GPU store so that the two storage locations remain independent.
    """

    def __init__(
        self,
        ram: ExpertStore,
        gpu: ExpertStore,
    ) -> None:
        self._ram = ram
        self._gpu = gpu

    def __call__(self, task: PrefetchTask) -> None:
        record = self._ram.get(task.expert_id)

        if record is None:
            raise KeyError(
                f"expert {task.expert_id!r} is not present in RAM storage"
            )

        gpu_record = ExpertRecord(
            expert_id=record.expert_id,
            location=ExpertLocation.RAM,
            payload=record.payload,
        )

        self._gpu.put(gpu_record)


def create_nvme_to_ram_handler(
    source: ExpertStore,
    target: ExpertStore,
) -> NvmeToRamHandler:
    return NvmeToRamHandler(
        ExpertStorePrefetchStorage(source),
        ExpertStorePrefetchRam(target),
    )


def create_ram_to_gpu_handler(
    source: ExpertStore,
    target: ExpertStore,
) -> RamToGpuHandler:
    return RamToGpuHandler(
        source,
        target,
    )


def create_storage_transfer_executor(
    source: ExpertStore,
    target: ExpertStore,
    default_handler: PrefetchTransferHandler,
    *,
    gpu: ExpertStore | None = None,
) -> PrefetchTransferExecutor:
    transfer = PrefetchTransferExecutor(default_handler)

    transfer.register(
        PrefetchSource.NVME,
        PrefetchTarget.RAM,
        create_nvme_to_ram_handler(source, target),
    )

    if gpu is not None:
        transfer.register(
            PrefetchSource.RAM,
            PrefetchTarget.GPU,
            create_ram_to_gpu_handler(target, gpu),
        )

    return transfer
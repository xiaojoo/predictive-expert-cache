from predictive_cache.prefetch.storage import (
    ExpertStorePrefetchRam,
    ExpertStorePrefetchStorage,
    NvmeToRamHandler,
)
from predictive_cache.prefetch.types import (
    PrefetchSource,
    PrefetchTarget,
    PrefetchTask,
)
from predictive_cache.storage.expert_record import ExpertRecord
from predictive_cache.storage.expert_store import ExpertLocation
from predictive_cache.storage.in_memory import InMemoryExpertStore
from predictive_cache.storage.nvme import NVMeExpertStore
from predictive_cache.storage.ram import RAMExpertStore


def test_expert_store_prefetch_storage_loads_payload() -> None:
    store = NVMeExpertStore()

    store.put(
        ExpertRecord(
            expert_id=7,
            location=ExpertLocation.NVME,
            payload="expert-7",
        )
    )

    storage = ExpertStorePrefetchStorage(store)

    assert storage.load(7) == "expert-7"


def test_expert_store_prefetch_storage_missing_expert() -> None:
    store = NVMeExpertStore()
    storage = ExpertStorePrefetchStorage(store)

    try:
        storage.load(7)
    except KeyError as exc:
        assert "expert 7" in str(exc)
    else:
        raise AssertionError("expected KeyError")


def test_expert_store_prefetch_ram_stores_record() -> None:
    store = RAMExpertStore()
    ram = ExpertStorePrefetchRam(store)

    ram.store(7, "expert-7")

    record = store.get(7)

    assert record is not None
    assert record.expert_id == 7
    assert record.location == ExpertLocation.RAM
    assert record.payload == "expert-7"


def test_nvme_to_ram_handler_uses_existing_storage() -> None:
    nvme = NVMeExpertStore()
    ram_store = RAMExpertStore()

    nvme.put(
        ExpertRecord(
            expert_id=7,
            location=ExpertLocation.NVME,
            payload="expert-7",
        )
    )

    source = ExpertStorePrefetchStorage(nvme)
    target = ExpertStorePrefetchRam(ram_store)

    handler = NvmeToRamHandler(source, target)

    task = PrefetchTask(
        expert_id=7,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
        priority=1.0,
    )

    handler(task)

    record = ram_store.get(7)

    assert record is not None
    assert record.location == ExpertLocation.RAM
    assert record.payload == "expert-7"


def test_nvme_to_ram_handler_composes_with_in_memory_store() -> None:
    store = InMemoryExpertStore()

    store.put(
        ExpertRecord(
            expert_id=7,
            location=ExpertLocation.NVME,
            payload="expert-7",
        )
    )

    source = ExpertStorePrefetchStorage(store)
    target = ExpertStorePrefetchRam(store)

    handler = NvmeToRamHandler(source, target)

    task = PrefetchTask(
        expert_id=7,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
        priority=1.0,
    )

    handler(task)

    record = store.get(7)

    assert record is not None
    assert record.location == ExpertLocation.RAM
    assert record.payload == "expert-7"
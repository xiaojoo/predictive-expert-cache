import pytest

from predictive_cache.storage.gpu import GPUExpertStore
from predictive_cache.storage.nvme import NVMeExpertStore
from predictive_cache.storage.ram import RAMExpertStore
from predictive_cache.storage import ExpertLocation, ExpertRecord, ExpertStore


@pytest.mark.parametrize(
    "store_cls, expected_location",
    [
        (GPUExpertStore, ExpertLocation.GPU),
        (RAMExpertStore, ExpertLocation.RAM),
        (NVMeExpertStore, ExpertLocation.NVME),
    ],
)
def test_backend_reports_its_location(store_cls, expected_location):
    store = store_cls()

    assert store.location("expert-a") == expected_location


@pytest.mark.parametrize(
    "store_cls",
    [
        GPUExpertStore,
        RAMExpertStore,
        NVMeExpertStore,
    ],
)

def test_backend_is_an_expert_store(store_cls):
    store = store_cls()

    assert hasattr(store, "load")
    assert hasattr(store, "unload")
    assert hasattr(store, "prefetch")
    assert hasattr(store, "location")

from predictive_cache.storage import ExpertStore


@pytest.mark.parametrize(
    "store_cls",
    [
        GPUExpertStore,
        RAMExpertStore,
        NVMeExpertStore,
    ],
)
def test_backend_inherits_expert_store(store_cls):
    assert issubclass(store_cls, ExpertStore)


@pytest.mark.parametrize(
    "store_cls",
    [
        GPUExpertStore,
        RAMExpertStore,
        NVMeExpertStore,
    ],
)
def test_backend_instance_is_expert_store(store_cls):
    store = store_cls()

    assert isinstance(store, ExpertStore)

@pytest.mark.parametrize(
    "store_cls",
    [
        GPUExpertStore,
        RAMExpertStore,
        NVMeExpertStore,
    ],
)
def test_backend_can_store_and_retrieve_expert_record(store_cls):
    store = store_cls()
    record = ExpertRecord(
        expert_id="expert-a",
        location=store.location("expert-a"),
        payload={"weights": "dummy"},
    )

    store.put(record)

    assert store.contains("expert-a")
    assert store.get("expert-a") is record


@pytest.mark.parametrize(
    "store_cls",
    [
        GPUExpertStore,
        RAMExpertStore,
        NVMeExpertStore,
    ],
)
def test_backend_can_remove_expert_record(store_cls):
    store = store_cls()
    record = ExpertRecord(
        expert_id="expert-a",
        location=store.location("expert-a"),
        payload={"weights": "dummy"},
    )

    store.put(record)
    store.remove("expert-a")

    assert not store.contains("expert-a")
    assert store.get("expert-a") is None
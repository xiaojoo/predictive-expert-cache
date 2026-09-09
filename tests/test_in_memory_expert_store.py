import pytest

from predictive_cache.storage import ExpertLocation
from predictive_cache.storage.in_memory import InMemoryExpertStore


def test_new_expert_starts_remote():
    store = InMemoryExpertStore()

    assert store.location("expert-a") == ExpertLocation.REMOTE


def test_prefetch_moves_remote_to_nvme():
    store = InMemoryExpertStore()

    store.prefetch("expert-a")

    assert store.location("expert-a") == ExpertLocation.NVME


def test_prefetch_moves_nvme_to_ram():
    store = InMemoryExpertStore()

    store.prefetch("expert-a")
    store.prefetch("expert-a")

    assert store.location("expert-a") == ExpertLocation.RAM


def test_load_moves_ram_to_gpu():
    store = InMemoryExpertStore()

    store.prefetch("expert-a")
    store.prefetch("expert-a")
    store.load("expert-a")

    assert store.location("expert-a") == ExpertLocation.GPU


def test_unload_moves_gpu_to_ram():
    store = InMemoryExpertStore()

    store.prefetch("expert-a")
    store.prefetch("expert-a")
    store.load("expert-a")
    store.unload("expert-a")

    assert store.location("expert-a") == ExpertLocation.RAM


def test_unload_moves_ram_to_nvme():
    store = InMemoryExpertStore()

    store.prefetch("expert-a")
    store.prefetch("expert-a")
    store.unload("expert-a")

    assert store.location("expert-a") == ExpertLocation.NVME


def test_unload_moves_nvme_to_remote():
    store = InMemoryExpertStore()

    store.prefetch("expert-a")
    store.unload("expert-a")

    assert store.location("expert-a") == ExpertLocation.REMOTE

def test_load_from_remote_is_rejected():
    store = InMemoryExpertStore()

    with pytest.raises(ValueError):
        store.load("expert-a")


def test_load_from_nvme_is_rejected():
    store = InMemoryExpertStore()
    store.prefetch("expert-a")

    with pytest.raises(ValueError):
        store.load("expert-a")


def test_load_from_gpu_is_idempotent():
    store = InMemoryExpertStore()
    store.prefetch("expert-a")
    store.prefetch("expert-a")
    store.load("expert-a")

    store.load("expert-a")

    assert store.location("expert-a") == ExpertLocation.GPU


def test_prefetch_from_ram_is_idempotent():
    store = InMemoryExpertStore()
    store.prefetch("expert-a")
    store.prefetch("expert-a")

    store.prefetch("expert-a")

    assert store.location("expert-a") == ExpertLocation.RAM


def test_prefetch_from_gpu_is_idempotent():
    store = InMemoryExpertStore()
    store.prefetch("expert-a")
    store.prefetch("expert-a")
    store.load("expert-a")

    store.prefetch("expert-a")

    assert store.location("expert-a") == ExpertLocation.GPU


def test_unload_from_remote_is_idempotent():
    store = InMemoryExpertStore()

    store.unload("expert-a")

    assert store.location("expert-a") == ExpertLocation.REMOTE
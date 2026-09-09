import pytest
import torch

from predictive_cache.storage import (
    ExpertLocation,
    ExpertRecord,
    GPUExpertStore,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available",
)


def test_gpu_store_put_moves_tensor_to_cuda():
    store = GPUExpertStore()

    payload = torch.randn(1024, 1024)

    record = ExpertRecord(
        expert_id=1,
        location=ExpertLocation.RAM,
        payload=payload,
    )

    store.put(record)

    stored = store.get(1)

    assert stored is not None
    assert stored.location == ExpertLocation.GPU
    assert isinstance(stored.payload, torch.Tensor)
    assert stored.payload.device == store.device
    assert stored.payload.device.type == "cuda"


def test_gpu_store_unload_moves_tensor_to_cpu():
    store = GPUExpertStore()

    payload = torch.randn(512, 512)

    store.put(
        ExpertRecord(
            expert_id=2,
            location=ExpertLocation.RAM,
            payload=payload,
        )
    )

    stored = store.get(2)

    assert stored is not None
    assert stored.payload.device.type == "cuda"

    store.unload(2)

    stored = store.get(2)

    assert stored is not None
    assert stored.location == ExpertLocation.RAM
    assert stored.payload.device.type == "cpu"


def test_gpu_store_load_moves_tensor_to_gpu():
    store = GPUExpertStore()

    payload = torch.randn(256, 256)

    store.put(
        ExpertRecord(
            expert_id=3,
            location=ExpertLocation.RAM,
            payload=payload,
        )
    )

    store.unload(3)
    store.load(3)

    stored = store.get(3)

    assert stored is not None
    assert stored.location == ExpertLocation.GPU
    assert stored.payload.device == store.device


def test_gpu_store_prefetch_is_gpu_load():
    store = GPUExpertStore()

    store.put(
        ExpertRecord(
            expert_id=4,
            location=ExpertLocation.RAM,
            payload=torch.randn(128, 128),
        )
    )

    store.unload(4)
    store.prefetch(4)

    stored = store.get(4)

    assert stored is not None
    assert stored.location == ExpertLocation.GPU
    assert stored.payload.device.type == "cuda"


def test_gpu_store_preserves_tensor_values():
    store = GPUExpertStore()

    payload = torch.randn(128, 128)

    store.put(
        ExpertRecord(
            expert_id=5,
            location=ExpertLocation.RAM,
            payload=payload,
        )
    )

    stored = store.get(5)

    assert stored is not None
    assert torch.equal(
        stored.payload.cpu(),
        payload,
    )

def test_gpu_store_put_preserves_record_identity():
    store = GPUExpertStore()

    record = ExpertRecord(
        expert_id=6,
        location=ExpertLocation.RAM,
        payload=torch.randn(64, 64),
    )

    store.put(record)

    assert store.get(6) is record
    assert record.location == ExpertLocation.GPU
    assert record.payload.device == store.device


def test_gpu_store_unload_and_load_preserve_values():
    store = GPUExpertStore()

    payload = torch.randn(64, 64)
    original = payload.clone()

    store.put(
        ExpertRecord(
            expert_id=7,
            location=ExpertLocation.RAM,
            payload=payload,
        )
    )

    store.unload(7)
    store.load(7)

    stored = store.get(7)

    assert stored is not None
    assert stored.location == ExpertLocation.GPU
    assert torch.equal(stored.payload.cpu(), original)


def test_gpu_store_non_tensor_payload_is_preserved():
    store = GPUExpertStore()

    payload = {"weights": "dummy", "metadata": [1, 2, 3]}

    record = ExpertRecord(
        expert_id=8,
        location=ExpertLocation.RAM,
        payload=payload,
    )

    store.put(record)

    stored = store.get(8)

    assert stored is record
    assert stored.location == ExpertLocation.GPU
    assert stored.payload is payload


def test_gpu_store_remove_clears_record():
    store = GPUExpertStore()

    store.put(
        ExpertRecord(
            expert_id=9,
            location=ExpertLocation.RAM,
            payload=torch.randn(32, 32),
        )
    )

    assert store.contains(9)

    store.remove(9)

    assert not store.contains(9)
    assert store.get(9) is None
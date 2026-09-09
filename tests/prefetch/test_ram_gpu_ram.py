import torch

from predictive_cache.storage.expert_record import ExpertRecord
from predictive_cache.storage.expert_store import ExpertLocation
from predictive_cache.storage.ram import RAMExpertStore
from predictive_cache.storage.gpu import GPUExpertStore


def test_ram_gpu_ram_preserves_tensor_and_location():
    ram = RAMExpertStore()
    gpu = GPUExpertStore()

    expert_id = 7
    original = torch.tensor(
        [1.0, 2.0, 3.0],
        dtype=torch.float32,
    )

    ram.put(
        ExpertRecord(
            expert_id=expert_id,
            location=ExpertLocation.RAM,
            payload=original,
        )
    )

    ram_record = ram.get(expert_id)
    assert ram_record is not None
    assert ram_record.location == ExpertLocation.RAM
    assert ram_record.payload.device.type == "cpu"

    gpu.put(
        ExpertRecord(
            expert_id=expert_id,
            location=ExpertLocation.RAM,
            payload=ram_record.payload,
        )
    )

    gpu_record = gpu.get(expert_id)
    assert gpu_record is not None
    assert gpu_record.location == ExpertLocation.GPU
    assert gpu_record.payload.device.type == "cuda"
    assert torch.equal(gpu_record.payload.cpu(), original)

    gpu.unload(expert_id)

    unloaded_record = gpu.get(expert_id)
    assert unloaded_record is not None
    assert unloaded_record.location == ExpertLocation.RAM
    assert unloaded_record.payload.device.type == "cpu"
    assert torch.equal(unloaded_record.payload, original)

    assert ram_record.location == ExpertLocation.RAM
    assert ram_record.payload.device.type == "cpu"
    assert torch.equal(ram_record.payload, original)
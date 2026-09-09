import torch

from predictive_cache.prefetch.storage import RamToGpuHandler
from predictive_cache.prefetch.types import PrefetchSource, PrefetchTarget, PrefetchTask
from predictive_cache.storage.expert_record import ExpertRecord
from predictive_cache.storage.expert_store import ExpertLocation
from predictive_cache.storage.ram import RAMExpertStore
from predictive_cache.storage.gpu import GPUExpertStore


def test_ram_to_gpu_moves_tensor():
    ram = RAMExpertStore()
    gpu = GPUExpertStore()

    tensor = torch.tensor([1.0, 2.0, 3.0])

    ram.put(
        ExpertRecord(
            expert_id=1,
            location=ExpertLocation.RAM,
            payload=tensor,
        )
    )

    handler = RamToGpuHandler(ram, gpu)

    task = PrefetchTask(
        expert_id=1,
        source=PrefetchSource.RAM,
        target=PrefetchTarget.GPU,
        priority=1.0,
        confidence=1.0,
        estimated_distance=0,
    )

    handler(task)

    ram_record = ram.get(1)
    gpu_record = gpu.get(1)

    assert ram_record is not None
    assert gpu_record is not None

    assert ram_record.location == ExpertLocation.RAM
    assert ram_record.payload.device.type == "cpu"

    assert gpu_record.location == ExpertLocation.GPU
    assert gpu_record.payload.device.type == "cuda"
    assert torch.equal(gpu_record.payload.cpu(), tensor)


def test_ram_to_gpu_missing_expert_fails():
    ram = RAMExpertStore()
    gpu = GPUExpertStore()

    handler = RamToGpuHandler(ram, gpu)

    task = PrefetchTask(
        expert_id=999,
        source=PrefetchSource.RAM,
        target=PrefetchTarget.GPU,
        priority=1.0,
        confidence=1.0,
        estimated_distance=0,
    )

    try:
        handler(task)
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")

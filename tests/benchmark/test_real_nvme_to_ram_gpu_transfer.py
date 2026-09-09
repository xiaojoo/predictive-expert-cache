from __future__ import annotations

import time
from pathlib import Path

import pytest
import torch

from predictive_cache.prefetch.storage import (
    create_nvme_to_ram_handler,
    create_ram_to_gpu_handler,
)
from predictive_cache.prefetch.types import (
    PrefetchSource,
    PrefetchTarget,
    PrefetchTask,
)
from predictive_cache.storage.expert_record import ExpertRecord
from predictive_cache.storage.expert_store import (
    ExpertLocation,
    ExpertStore,
)
from predictive_cache.storage.gpu import GPUExpertStore
from predictive_cache.storage.nvme import NVMeExpertStore


class MemoryExpertStore(ExpertStore):
    """Test-only CPU/RAM expert store."""

    def __init__(self) -> None:
        self._records: dict[int, ExpertRecord] = {}

    def put(self, record: ExpertRecord) -> None:
        record.location = ExpertLocation.RAM
        self._records[record.expert_id] = record

    def get(self, expert_id: int) -> ExpertRecord | None:
        return self._records.get(expert_id)

    def contains(self, expert_id: int) -> bool:
        return expert_id in self._records

    def remove(self, expert_id: int) -> None:
        self._records.pop(expert_id, None)

    def load(self, expert_id: int) -> None:
        raise NotImplementedError

    def unload(self, expert_id: int) -> None:
        raise NotImplementedError

    def prefetch(self, expert_id: int) -> None:
        raise NotImplementedError

    def location(self, expert_id: int) -> ExpertLocation:
        record = self._records.get(expert_id)

        if record is None:
            return ExpertLocation.RAM

        return record.location


def test_real_nvme_to_ram_to_gpu_transfer() -> None:
    print()
    print("7-O-2 Real NVMe -> RAM -> GPU Transfer")
    print("---------------------------------------")

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    expert_id = 26
    element_count = 2 * 1024 * 1024

    payload = torch.arange(
        element_count,
        dtype=torch.float32,
        device="cpu",
    )

    payload_bytes = (
        payload.numel() * payload.element_size()
    )

    nvme_dir = (
        Path.cwd() / ".pytest_nvme_gpu_benchmark"
    )

    nvme = NVMeExpertStore(nvme_dir)
    ram = MemoryExpertStore()
    gpu = GPUExpertStore("cuda")

    try:
        nvme.put(
            ExpertRecord(
                expert_id=expert_id,
                location=ExpertLocation.NVME,
                payload=payload,
            )
        )

        nvme_path = nvme_dir / f"{expert_id}.pkl"

        assert nvme_path.exists()
        assert nvme_path.stat().st_size > 0

        # ---------------------------------------------------------
        # NVMe -> RAM
        # ---------------------------------------------------------

        nvme_to_ram = create_nvme_to_ram_handler(
            nvme,
            ram,
        )

        task_nvme_ram = PrefetchTask(
            expert_id=expert_id,
            source=PrefetchSource.NVME,
            target=PrefetchTarget.RAM,
        )

        start = time.perf_counter()

        nvme_to_ram(task_nvme_ram)

        nvme_elapsed = (
            time.perf_counter() - start
        )

        ram_record = ram.get(expert_id)

        assert ram_record is not None
        assert (
            ram_record.location
            == ExpertLocation.RAM
        )
        assert isinstance(
            ram_record.payload,
            torch.Tensor,
        )
        assert (
            ram_record.payload.device.type
            == "cpu"
        )
        assert torch.equal(
            ram_record.payload,
            payload,
        )

        # ---------------------------------------------------------
        # RAM -> GPU
        # ---------------------------------------------------------

        ram_to_gpu = create_ram_to_gpu_handler(
            ram,
            gpu,
        )

        task_ram_gpu = PrefetchTask(
            expert_id=expert_id,
            source=PrefetchSource.RAM,
            target=PrefetchTarget.GPU,
        )

        torch.cuda.synchronize()

        start = time.perf_counter()

        ram_to_gpu(task_ram_gpu)

        torch.cuda.synchronize()

        gpu_elapsed = (
            time.perf_counter() - start
        )

        gpu_record = gpu.get(expert_id)

        assert gpu_record is not None
        assert (
            gpu_record.location
            == ExpertLocation.GPU
        )
        assert isinstance(
            gpu_record.payload,
            torch.Tensor,
        )
        assert (
            gpu_record.payload.device.type
            == "cuda"
        )
        assert (
            gpu_record.payload.device
            == gpu.device
        )

        # End-to-end data integrity.
        assert torch.equal(
            gpu_record.payload.cpu(),
            payload,
        )

        # RAM must remain independently resident.
        assert ram.contains(expert_id)
        assert gpu.contains(expert_id)

        print(f"expert:                 {expert_id}")
        print(
            f"payload bytes:          "
            f"{payload_bytes:,}"
        )
        print(
            f"NVMe file bytes:        "
            f"{nvme_path.stat().st_size:,}"
        )
        print(
            f"NVMe -> RAM latency:    "
            f"{nvme_elapsed * 1000:.3f} ms"
        )
        print(
            f"RAM resident:           "
            f"{ram.contains(expert_id)}"
        )
        print(
            f"GPU device:             "
            f"{gpu_record.payload.device}"
        )
        print(
            f"RAM -> GPU latency:     "
            f"{gpu_elapsed * 1000:.3f} ms"
        )
        print(
            f"GPU resident:           "
            f"{gpu.contains(expert_id)}"
        )

    finally:
        gpu.remove(expert_id)
        ram.remove(expert_id)
        nvme.remove(expert_id)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

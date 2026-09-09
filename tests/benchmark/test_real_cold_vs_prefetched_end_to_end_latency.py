from __future__ import annotations

import statistics
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


def _median(values: list[float]) -> float:
    return statistics.median(values)


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    index = max(
        0,
        min(
            len(ordered) - 1,
            int(len(ordered) * 0.95) - 1,
        ),
    )
    return ordered[index]


def test_real_cold_vs_prefetched_end_to_end_latency() -> None:
    print()
    print("7-O-3 Real Cold vs Prefetched End-to-End Latency")
    print("------------------------------------------------")

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    expert_id = 26
    element_count = 2 * 1024 * 1024
    samples = 20

    payload = torch.arange(
        element_count,
        dtype=torch.float32,
        device="cpu",
    )

    nvme_dir = (
        Path.cwd() / ".pytest_nvme_latency_benchmark"
    )

    nvme = NVMeExpertStore(nvme_dir)
    ram = MemoryExpertStore()
    gpu = GPUExpertStore("cuda")

    nvme_to_ram = create_nvme_to_ram_handler(
        nvme,
        ram,
    )

    ram_to_gpu = create_ram_to_gpu_handler(
        ram,
        gpu,
    )

    task_nvme_ram = PrefetchTask(
        expert_id=expert_id,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
    )

    task_ram_gpu = PrefetchTask(
        expert_id=expert_id,
        source=PrefetchSource.RAM,
        target=PrefetchTarget.GPU,
    )

    try:
        nvme.put(
            ExpertRecord(
                expert_id=expert_id,
                location=ExpertLocation.NVME,
                payload=payload,
            )
        )

        assert nvme.contains(expert_id)

        # Warm CUDA runtime once. Do not include initialization
        # overhead in the benchmark.
        warmup = torch.empty(
            1,
            device="cuda",
        )
        torch.cuda.synchronize()
        del warmup

        cold_latencies: list[float] = []
        prefetched_latencies: list[float] = []

        # ---------------------------------------------------------
        # Cold path
        # ---------------------------------------------------------

        for _ in range(samples):
            gpu.remove(expert_id)
            ram.remove(expert_id)

            assert not ram.contains(expert_id)
            assert not gpu.contains(expert_id)

            start = time.perf_counter()

            nvme_to_ram(task_nvme_ram)
            ram_to_gpu(task_ram_gpu)

            torch.cuda.synchronize()

            elapsed = (
                time.perf_counter() - start
            )

            cold_latencies.append(
                elapsed * 1000.0
            )

            record = gpu.get(expert_id)

            assert record is not None
            assert record.location == ExpertLocation.GPU
            assert isinstance(
                record.payload,
                torch.Tensor,
            )
            assert record.payload.device.type == "cuda"
            assert torch.equal(
                record.payload.cpu(),
                payload,
            )

        # ---------------------------------------------------------
        # Prefetched path
        # ---------------------------------------------------------

        for _ in range(samples):
            gpu.remove(expert_id)
            ram.remove(expert_id)

            assert not ram.contains(expert_id)
            assert not gpu.contains(expert_id)

            # Prefetch happens before the demand measurement.
            nvme_to_ram(task_nvme_ram)
            ram_to_gpu(task_ram_gpu)

            torch.cuda.synchronize()

            assert ram.contains(expert_id)
            assert gpu.contains(expert_id)

            start = time.perf_counter()

            record = gpu.get(expert_id)

            assert record is not None

            tensor = record.payload

            assert isinstance(
                tensor,
                torch.Tensor,
            )

            # Demand consumes the already-prefetched GPU
            # resident tensor.
            torch.cuda.synchronize()

            elapsed = (
                time.perf_counter() - start
            )

            prefetched_latencies.append(
                elapsed * 1000.0
            )

            assert tensor.device.type == "cuda"
            assert torch.equal(
                tensor.cpu(),
                payload,
            )

        cold_median = _median(
            cold_latencies
        )
        cold_p95 = _p95(
            cold_latencies
        )

        prefetched_median = _median(
            prefetched_latencies
        )
        prefetched_p95 = _p95(
            prefetched_latencies
        )

        speedup = (
            cold_median / prefetched_median
            if prefetched_median > 0
            else float("inf")
        )

        print(
            f"payload bytes:             "
            f"{payload.numel() * payload.element_size():,}"
        )
        print(
            f"samples:                   {samples}"
        )
        print(
            f"cold median:               "
            f"{cold_median:.3f} ms"
        )
        print(
            f"cold p95:                  "
            f"{cold_p95:.3f} ms"
        )
        print(
            f"prefetched median:         "
            f"{prefetched_median:.3f} ms"
        )
        print(
            f"prefetched p95:            "
            f"{prefetched_p95:.3f} ms"
        )
        print(
            f"median speedup:            "
            f"{speedup:.2f}x"
        )

        assert len(cold_latencies) == samples
        assert len(prefetched_latencies) == samples
        assert all(
            latency >= 0
            for latency in cold_latencies
        )
        assert all(
            latency >= 0
            for latency in prefetched_latencies
        )

    finally:
        gpu.remove(expert_id)
        ram.remove(expert_id)
        nvme.remove(expert_id)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

from __future__ import annotations

import time
from pathlib import Path

from predictive_cache.prefetch.ram import InMemoryPrefetchRam
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
from predictive_cache.storage.nvme import NVMeExpertStore


def test_real_nvme_to_ram_transfer() -> None:
    print()
    print("7-O-1 Real NVMe -> RAM Transfer")
    print("--------------------------------")

    payload_size = 8 * 1024 * 1024
    expert_id = 26

    payload = bytes(
        (index % 251 for index in range(payload_size))
    )

    nvme_dir = Path.cwd() / ".pytest_nvme_benchmark"
    nvme_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    nvme = NVMeExpertStore(nvme_dir)
    ram = InMemoryPrefetchRam()

    record = ExpertRecord(
        expert_id=expert_id,
        location=ExpertLocation.NVME,
        payload=payload,
    )

    nvme.put(record)

    path = nvme_dir / f"{expert_id}.pkl"

    assert path.exists()
    assert path.stat().st_size > payload_size

    stored = nvme.get(expert_id)

    assert stored is not None
    assert stored.expert_id == expert_id
    assert stored.location == ExpertLocation.NVME
    assert stored.payload == payload

    handler = NvmeToRamHandler(
        ExpertStorePrefetchStorage(nvme),
        ram,
    )

    task = PrefetchTask(
        expert_id=expert_id,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
    )

    start = time.perf_counter()

    handler(task)

    elapsed = time.perf_counter() - start

    assert ram.contains(expert_id)

    transferred = ram.get(expert_id)

    assert transferred == payload

    print(f"expert:             {expert_id}")
    print(f"payload bytes:      {payload_size:,}")
    print(f"NVMe file bytes:    {path.stat().st_size:,}")
    print(f"NVMe file:          {path}")
    print(f"RAM resident:       {ram.contains(expert_id)}")
    print(f"transfer latency:   {elapsed * 1000:.3f} ms")

    nvme.remove(expert_id)

    assert not nvme.contains(expert_id)
    assert not path.exists()

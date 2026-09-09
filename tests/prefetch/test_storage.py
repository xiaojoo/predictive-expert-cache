from predictive_cache.prefetch import (
    InMemoryPrefetchRam,
    NvmeToRamHandler,
    PrefetchEngine,
    PrefetchSource,
    PrefetchTarget,
    PrefetchTask,
    PrefetchTransferExecutor,
)

class FakeStorage:
    def __init__(self) -> None:
        self.loaded: list[int] = []

    def load(self, expert_id: int) -> object:
        self.loaded.append(expert_id)
        return f"expert-{expert_id}"


class FakeRam:
    def __init__(self) -> None:
        self.stored: dict[int, object] = {}

    def store(self, expert_id: int, data: object) -> None:
        self.stored[expert_id] = data


def test_nvme_to_ram_handler_loads_and_stores():
    storage = FakeStorage()
    ram = FakeRam()

    handler = NvmeToRamHandler(
        storage,
        ram,
    )

    class Task:
        expert_id = 7

    handler(Task())

    assert storage.loaded == [7]
    assert ram.stored == {
        7: "expert-7",
    }


def test_nvme_to_ram_handler_preserves_loaded_data():
    storage = FakeStorage()
    ram = FakeRam()

    handler = NvmeToRamHandler(
        storage,
        ram,
    )

    class Task:
        expert_id = 21

    handler(Task())

    assert ram.stored[21] == storage.load(21)

def test_nvme_to_ram_full_prefetch_chain():
    class Storage:
        def load(self, expert_id: int) -> object:
            return {
                "expert_id": expert_id,
                "payload": f"expert-{expert_id}",
            }

    storage = Storage()
    ram = InMemoryPrefetchRam()

    handler = NvmeToRamHandler(
        storage,
        ram,
    )

    transfer = PrefetchTransferExecutor(
        lambda task: None,
    )

    transfer.register(
        PrefetchSource.NVME,
        PrefetchTarget.RAM,
        handler,
    )

    engine = PrefetchEngine(
        lambda task: None,
        transfer_executor=transfer,
        num_workers=1,
    )

    task = PrefetchTask(
        expert_id=42,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
        priority=0.9,
        confidence=0.95,
        estimated_distance=1,
    )

    engine.start()

    assert engine.submit(task) is True

    engine.stop()

    assert ram.contains(42) is True
    assert ram.get(42) == {
        "expert_id": 42,
        "payload": "expert-42",
    }

    assert engine.task_status(42).value == "completed"

def test_nvme_to_ram_prefetch_runs_concurrently_with_multiple_workers():
    import threading

    from predictive_cache.prefetch.storage import (
        create_nvme_to_ram_handler,
    )
    from predictive_cache.prefetch.types import (
        PrefetchStatus,
    )
    from predictive_cache.storage.expert_record import ExpertRecord
    from predictive_cache.storage.expert_store import ExpertLocation
    from predictive_cache.storage.nvme import NVMeExpertStore
    from predictive_cache.storage.ram import RAMExpertStore

    nvme = NVMeExpertStore()
    ram = RAMExpertStore()

    experts = {
        101: "expert-101",
        102: "expert-102",
    }

    for expert_id, payload in experts.items():
        nvme.put(
            ExpertRecord(
                expert_id=expert_id,
                location=ExpertLocation.NVME,
                payload=payload,
            )
        )

    storage_handler = create_nvme_to_ram_handler(
        nvme,
        ram,
    )

    entered = []
    entered_lock = threading.Lock()
    both_workers_started = threading.Event()
    release_workers = threading.Event()

    def concurrent_handler(task):
        with entered_lock:
            entered.append(task.expert_id)

            if len(entered) == 2:
                both_workers_started.set()

        assert release_workers.wait(timeout=5)

        storage_handler(task)

    transfer = PrefetchTransferExecutor(
        lambda task: None,
    )

    transfer.register(
        PrefetchSource.NVME,
        PrefetchTarget.RAM,
        concurrent_handler,
    )

    engine = PrefetchEngine(
        lambda task: None,
        transfer_executor=transfer,
        num_workers=2,
    )

    tasks = [
        PrefetchTask(
            expert_id=101,
            source=PrefetchSource.NVME,
            target=PrefetchTarget.RAM,
            priority=0.9,
        ),
        PrefetchTask(
            expert_id=102,
            source=PrefetchSource.NVME,
            target=PrefetchTarget.RAM,
            priority=0.8,
        ),
    ]

    engine.start()

    try:
        assert engine.submit(tasks[0]) is True
        assert engine.submit(tasks[1]) is True

        assert both_workers_started.wait(timeout=5)

        with entered_lock:
            assert set(entered) == {101, 102}

        release_workers.set()
    finally:
        release_workers.set()
        engine.stop()

    results = engine.results()

    assert len(results) == 2
    assert {
        result.expert_id
        for result in results
    } == {101, 102}

    assert all(
        result.status == PrefetchStatus.COMPLETED
        for result in results
    )

    assert ram.get(101).payload == "expert-101"
    assert ram.get(102).payload == "expert-102"

    assert nvme.get(101).payload == "expert-101"
    assert nvme.get(102).payload == "expert-102"
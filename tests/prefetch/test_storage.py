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
from predictive_cache.prefetch import NvmeToRamHandler


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
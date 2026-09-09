import time

from predictive_cache.prefetch import (
    PrefetchEngine,
    PrefetchSource,
    PrefetchStatus,
    PrefetchTarget,
    PrefetchTask,
    PrefetchTransferExecutor,
    NvmeToRamHandler,
    create_storage_transfer_executor
)


def test_engine_executes_task():
    executed = []

    def loader(task):
        executed.append(task.expert_id)

    engine = PrefetchEngine(loader)

    engine.start()

    try:
        task = PrefetchTask(
            expert_id=41,
            source=PrefetchSource.NVME,
            target=PrefetchTarget.RAM,
            priority=1.0,
            confidence=0.95,
        )

        assert engine.submit(task)

        deadline = time.time() + 2.0

        while time.time() < deadline:
            if 41 in executed:
                break

            time.sleep(0.01)

        assert executed == [41]

        results = engine.results()

        assert len(results) == 1
        assert results[0].expert_id == 41
        assert results[0].status == PrefetchStatus.COMPLETED

    finally:
        engine.stop()


def test_engine_failure_is_recorded():
    def loader(task):
        raise RuntimeError("load failed")

    engine = PrefetchEngine(loader)

    engine.start()

    try:
        task = PrefetchTask(
            expert_id=41,
            source=PrefetchSource.NVME,
            target=PrefetchTarget.RAM,
            priority=1.0,
        )

        assert engine.submit(task)

        deadline = time.time() + 2.0

        while time.time() < deadline:
            results = engine.results()

            if results:
                break

            time.sleep(0.01)

        assert len(results) == 1

        result = results[0]

        assert result.status == PrefetchStatus.FAILED
        assert result.error == "load failed"

    finally:
        engine.stop()

def test_engine_preserves_transfer_metadata():
    received = []

    def loader(task: PrefetchTask) -> None:
        received.append(task)

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )

    task = PrefetchTask(
        expert_id=41,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
        priority=0.8,
        confidence=0.95,
        estimated_distance=3,
    )

    engine.start()

    assert engine.submit(task) is True

    engine.stop()

    assert len(received) == 1

    executed = received[0]

    assert executed.expert_id == 41
    assert executed.source == PrefetchSource.NVME
    assert executed.target == PrefetchTarget.RAM
    assert executed.priority == 0.8
    assert executed.confidence == 0.95
    assert executed.estimated_distance == 3

def test_engine_uses_transfer_route():
    received: list[int] = []

    def loader(task: PrefetchTask) -> None:
        received.append(task.expert_id)

    transfer = PrefetchTransferExecutor(loader)

    engine = PrefetchEngine(
        loader,
        transfer_executor=transfer,
        num_workers=1,
    )

    task = PrefetchTask(
        expert_id=41,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
        priority=0.8,
        confidence=0.95,
        estimated_distance=3,
    )

    route_calls: list[int] = []

    def nvme_to_ram(task: PrefetchTask) -> None:
        route_calls.append(task.expert_id)

    transfer.register(
        PrefetchSource.NVME,
        PrefetchTarget.RAM,
        nvme_to_ram,
    )

    engine.start()

    assert engine.submit(task) is True

    engine.stop()

    assert route_calls == [41]
    assert received == []

def test_engine_executes_nvme_to_ram_handler():
    loaded: list[int] = []
    stored: dict[int, object] = {}

    class Storage:
        def load(self, expert_id: int) -> object:
            loaded.append(expert_id)
            return f"expert-{expert_id}"

    class Ram:
        def store(self, expert_id: int, data: object) -> None:
            stored[expert_id] = data

    storage = Storage()
    ram = Ram()

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
        expert_id=31,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
        priority=0.8,
        confidence=0.9,
        estimated_distance=2,
    )

    engine.start()

    assert engine.submit(task) is True

    engine.stop()

    assert loaded == [31]
    assert stored == {
        31: "expert-31",
    }

def test_engine_executes_nvme_to_ram_with_expert_stores():
    from predictive_cache.storage.expert_record import ExpertRecord
    from predictive_cache.storage.expert_store import ExpertLocation
    from predictive_cache.storage.nvme import NVMeExpertStore
    from predictive_cache.storage.ram import RAMExpertStore
    from predictive_cache.prefetch.storage import (
        ExpertStorePrefetchRam,
        ExpertStorePrefetchStorage,
    )

    nvme = NVMeExpertStore()
    ram = RAMExpertStore()

    nvme.put(
        ExpertRecord(
            expert_id=51,
            location=ExpertLocation.NVME,
            payload="expert-51",
        )
    )

    transfer = create_storage_transfer_executor(
        nvme,
        ram,
        lambda task: None,
    )

    engine = PrefetchEngine(
        lambda task: None,
        transfer_executor=transfer,
        num_workers=1,
    )

    task = PrefetchTask(
        expert_id=51,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
        priority=0.9,
        confidence=0.95,
        estimated_distance=2,
    )

    engine.start()

    assert engine.submit(task) is True

    engine.stop()

    result = engine.results()

    assert len(result) == 1
    assert result[0].expert_id == 51
    assert result[0].status == PrefetchStatus.COMPLETED

    nvme_record = nvme.get(51)
    ram_record = ram.get(51)

    assert nvme_record is not None
    assert nvme_record.location == ExpertLocation.NVME
    assert nvme_record.payload == "expert-51"

    assert ram_record is not None
    assert ram_record.expert_id == 51
    assert ram_record.location == ExpertLocation.RAM
    assert ram_record.payload == "expert-51"

def test_engine_reports_storage_backed_nvme_to_ram_failure():
    from predictive_cache.prefetch.storage import (
        create_storage_transfer_executor,
    )
    from predictive_cache.storage.nvme import NVMeExpertStore
    from predictive_cache.storage.ram import RAMExpertStore

    nvme = NVMeExpertStore()
    ram = RAMExpertStore()

    def loader(task):
        raise AssertionError(
            "default loader must not handle NVMe -> RAM"
        )

    transfer = create_storage_transfer_executor(
        nvme,
        ram,
        loader,
    )

    engine = PrefetchEngine(
        loader,
        transfer_executor=transfer,
        num_workers=1,
    )

    task = PrefetchTask(
        expert_id=999,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
        priority=0.9,
        confidence=0.9,
        estimated_distance=1,
    )

    engine.start()

    assert engine.submit(task) is True

    engine.stop()

    results = engine.results()

    assert len(results) == 1

    result = results[0]

    assert result.expert_id == 999
    assert result.status == PrefetchStatus.FAILED
    assert result.error is not None
    assert "expert 999" in result.error

    assert nvme.get(999) is None
    assert ram.get(999) is None
import time

from predictive_cache.prefetch import (
    PrefetchEngine,
    PrefetchSource,
    PrefetchStatus,
    PrefetchTarget,
    PrefetchTask,
    PrefetchTransferExecutor,
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
import time

from predictive_cache.prefetch import (
    PrefetchEngine,
    PrefetchSource,
    PrefetchStatus,
    PrefetchTarget,
    PrefetchTask,
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

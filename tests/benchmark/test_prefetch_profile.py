from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns

from predictive_cache.prefetch.storage import NvmeToRamHandler
from predictive_cache.prefetch.types import (
    PrefetchSource,
    PrefetchTarget,
    PrefetchTask,
)
from predictive_cache.storage.expert_record import ExpertRecord
from predictive_cache.storage.expert_store import ExpertLocation
from predictive_cache.storage.in_memory import InMemoryExpertStore
import threading
import time

@dataclass(frozen=True)
class ProfileResult:
    name: str
    cycles: int
    mean_us: float
    median_us: float
    p95_us: float
    min_us: float
    max_us: float
    throughput: float


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower

    return ordered[lower] + (
        ordered[upper] - ordered[lower]
    ) * fraction


def _summarize(name: str, samples_ns: list[int]) -> ProfileResult:
    samples_us = [value / 1_000.0 for value in samples_ns]

    total_seconds = sum(samples_ns) / 1_000_000_000.0

    return ProfileResult(
        name=name,
        cycles=len(samples_ns),
        mean_us=sum(samples_us) / len(samples_us),
        median_us=_percentile(samples_us, 0.50),
        p95_us=_percentile(samples_us, 0.95),
        min_us=min(samples_us),
        max_us=max(samples_us),
        throughput=len(samples_ns) / total_seconds,
    )


def _print_result(result: ProfileResult) -> None:
    print()
    print(result.name)
    print("-" * len(result.name))
    print(f"cycles:     {result.cycles}")
    print(f"mean:       {result.mean_us:.3f} us")
    print(f"median:     {result.median_us:.3f} us")
    print(f"p95:        {result.p95_us:.3f} us")
    print(f"min:        {result.min_us:.3f} us")
    print(f"max:        {result.max_us:.3f} us")
    print(f"throughput: {result.throughput:.3f} ops/s")


def _make_task(expert_id: int) -> PrefetchTask:
    return PrefetchTask(
        expert_id=expert_id,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
    )


def _make_record(expert_id: int) -> ExpertRecord:
    return ExpertRecord(
        expert_id=expert_id,
        location=ExpertLocation.NVME,
        payload=f"expert-{expert_id}".encode(),
    )


def _make_stores(
    cycles: int,
) -> tuple[InMemoryExpertStore, InMemoryExpertStore]:
    source = InMemoryExpertStore()
    target = InMemoryExpertStore()

    for expert_id in range(cycles):
        source.put(_make_record(expert_id))

    return source, target


def _benchmark_handler(cycles: int = 10_000) -> ProfileResult:
    source, target = _make_stores(cycles)

    from predictive_cache.prefetch.storage import (
        ExpertStorePrefetchRam,
        ExpertStorePrefetchStorage,
    )

    handler = NvmeToRamHandler(
        storage=ExpertStorePrefetchStorage(source),
        ram=ExpertStorePrefetchRam(target),
    )

    samples: list[int] = []

    for expert_id in range(cycles):
        task = _make_task(expert_id)

        started = perf_counter_ns()
        handler(task)
        samples.append(perf_counter_ns() - started)

    return _summarize(
        "NvmeToRamHandler direct execution",
        samples,
    )


def _benchmark_source_get(cycles: int = 10_000) -> ProfileResult:
    source, _ = _make_stores(cycles)

    samples: list[int] = []

    for expert_id in range(cycles):
        started = perf_counter_ns()
        source.get(expert_id)
        samples.append(perf_counter_ns() - started)

    return _summarize(
        "InMemoryExpertStore.get",
        samples,
    )


def _benchmark_target_put(cycles: int = 10_000) -> ProfileResult:
    source, target = _make_stores(cycles)

    samples: list[int] = []

    for expert_id in range(cycles):
        record = source.get(expert_id)

        started = perf_counter_ns()
        target.put(record)
        samples.append(perf_counter_ns() - started)

    return _summarize(
        "InMemoryExpertStore.put",
        samples,
    )


def test_profile_nvme_to_ram_handler_10000_cycles() -> None:
    result = _benchmark_handler(cycles=10_000)

    _print_result(result)

    assert result.cycles == 10_000
    assert result.throughput > 0.0


def test_profile_source_get_10000_cycles() -> None:
    result = _benchmark_source_get(cycles=10_000)

    _print_result(result)

    assert result.cycles == 10_000
    assert result.throughput > 0.0


def test_profile_target_put_10000_cycles() -> None:
    result = _benchmark_target_put(cycles=10_000)

    _print_result(result)

    assert result.cycles == 10_000
    assert result.throughput > 0.0

from predictive_cache.prefetch.transfer import PrefetchTransferExecutor


class _DirectExecutor:
    def __init__(self, handler) -> None:
        self._handler = handler

    def execute(self, task):
        return self._handler(task)


def _benchmark_direct_executor(cycles: int = 10_000) -> ProfileResult:
    source, target = _make_stores(cycles)

    from predictive_cache.prefetch.storage import (
        ExpertStorePrefetchRam,
        ExpertStorePrefetchStorage,
    )

    handler = NvmeToRamHandler(
        storage=ExpertStorePrefetchStorage(source),
        ram=ExpertStorePrefetchRam(target),
    )

    executor = _DirectExecutor(handler)

    samples: list[int] = []

    for expert_id in range(cycles):
        task = _make_task(expert_id)

        started = perf_counter_ns()
        executor.execute(task)
        samples.append(perf_counter_ns() - started)

    return _summarize(
        "Direct executor -> handler",
        samples,
    )


def test_profile_direct_executor_10000_cycles() -> None:
    result = _benchmark_direct_executor(cycles=10_000)

    _print_result(result)

    assert result.cycles == 10_000
    assert result.throughput > 0.0

@dataclass
class _EngineTiming:
    execute_started_ns: int = 0
    execute_finished_ns: int = 0


class _TimedExecutor:
    def __init__(self, handler) -> None:
        self._handler = handler
        self._lock = threading.Lock()
        self._timing = _EngineTiming()
        self.started = threading.Event()
        self.finished = threading.Event()

    def reset(self) -> None:
        with self._lock:
            self._timing = _EngineTiming()

        self.started.clear()
        self.finished.clear()

    def execute(self, task) -> None:
        started = perf_counter_ns()

        with self._lock:
            self._timing.execute_started_ns = started

        self.started.set()

        try:
            self._handler(task)
        finally:
            finished = perf_counter_ns()

            with self._lock:
                self._timing.execute_finished_ns = finished

            self.finished.set()

    def timing(self) -> _EngineTiming:
        with self._lock:
            return _EngineTiming(
                execute_started_ns=self._timing.execute_started_ns,
                execute_finished_ns=self._timing.execute_finished_ns,
            )


def _benchmark_engine_breakdown(
    cycles: int = 10_000,
) -> dict[str, ProfileResult]:
    source, target = _make_stores(cycles)

    from predictive_cache.prefetch.storage import (
        ExpertStorePrefetchRam,
        ExpertStorePrefetchStorage,
    )

    handler = NvmeToRamHandler(
        storage=ExpertStorePrefetchStorage(source),
        ram=ExpertStorePrefetchRam(target),
    )

    timed_executor = _TimedExecutor(handler)

    from predictive_cache.prefetch.engine import PrefetchEngine
    from predictive_cache.prefetch.types import PrefetchStatus

    engine = PrefetchEngine(
        loader=lambda task: None,
        num_workers=1,
        transfer_executor=timed_executor,
    )

    engine.start()

    submit_samples: list[int] = []
    queue_samples: list[int] = []
    execute_samples: list[int] = []
    completion_samples: list[int] = []
    total_samples: list[int] = []

    try:
        for expert_id in range(cycles):
            task = _make_task(expert_id)

            timed_executor.reset()

            submit_started = perf_counter_ns()

            accepted = engine.submit(task)

            submit_finished = perf_counter_ns()

            assert accepted is True

            submit_samples.append(
                submit_finished - submit_started
            )

            # Worker has actually started _execute().
            assert timed_executor.started.wait(timeout=5.0)

            timing = timed_executor.timing()

            queue_samples.append(
                timing.execute_started_ns - submit_finished
            )

            # Wait until transfer executor itself has returned.
            assert timed_executor.finished.wait(timeout=5.0)

            timing = timed_executor.timing()

            execute_samples.append(
                timing.execute_finished_ns
                - timing.execute_started_ns
            )

            # Now measure the remaining Engine completion path.
            completion_started = perf_counter_ns()

            while (
                engine.task_status(expert_id)
                != PrefetchStatus.COMPLETED
            ):
                time.sleep(0)

            completion_finished = perf_counter_ns()

            completion_samples.append(
                completion_finished
                - timing.execute_finished_ns
            )

            total_samples.append(
                completion_finished - submit_started
            )

    finally:
        engine.stop()

    return {
        "submit": _summarize(
            "Engine.submit()",
            submit_samples,
        ),
        "queue": _summarize(
            "submit return -> worker execute start",
            queue_samples,
        ),
        "execute": _summarize(
            "worker execute()",
            execute_samples,
        ),
        "completion": _summarize(
            "execute finish -> COMPLETED observed",
            completion_samples,
        ),
        "total": _summarize(
            "submit -> COMPLETED observed",
            total_samples,
        ),
    }


def test_profile_engine_breakdown_10000_cycles() -> None:
    results = _benchmark_engine_breakdown(cycles=10_000)

    for result in results.values():
        _print_result(result)

        assert result.cycles == 10_000
        assert result.throughput > 0.0
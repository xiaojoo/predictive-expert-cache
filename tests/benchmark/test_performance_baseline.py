from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from statistics import mean, median
from typing import Callable

from predictive_cache.prefetch.engine import PrefetchEngine
from predictive_cache.prefetch.pipeline import PrefetchPipeline
from predictive_cache.prefetch.storage import (
    create_storage_transfer_executor,
)
from predictive_cache.prefetch.types import (
    PrefetchSource,
    PrefetchStatus,
    PrefetchTarget,
    PrefetchTask,
)
from predictive_cache.scheduler import ExpertScheduler, PrefetchRequest
from predictive_cache.storage.expert_record import ExpertRecord
from predictive_cache.storage.expert_store import ExpertLocation
from predictive_cache.storage.in_memory import InMemoryExpertStore


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    cycles: int
    mean_us: float
    median_us: float
    p95_us: float
    min_us: float
    max_us: float
    throughput_ops: float


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        raise ValueError("cannot calculate percentile of empty sample set")

    ordered = sorted(values)

    if len(ordered) == 1:
        return float(ordered[0])

    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = lower + 1

    if upper >= len(ordered):
        return float(ordered[-1])

    weight = rank - lower

    return (
        ordered[lower]
        + (ordered[upper] - ordered[lower]) * weight
    )


def _summarize(
    name: str,
    samples_ns: list[int],
) -> BenchmarkResult:
    if not samples_ns:
        raise ValueError("benchmark produced no samples")

    cycles = len(samples_ns)

    total_seconds = sum(samples_ns) / 1_000_000_000.0

    return BenchmarkResult(
        name=name,
        cycles=cycles,
        mean_us=mean(samples_ns) / 1_000.0,
        median_us=median(samples_ns) / 1_000.0,
        p95_us=_percentile(samples_ns, 0.95) / 1_000.0,
        min_us=min(samples_ns) / 1_000.0,
        max_us=max(samples_ns) / 1_000.0,
        throughput_ops=(
            cycles / total_seconds
            if total_seconds > 0.0
            else float("inf")
        ),
    )


def _print_result(result: BenchmarkResult) -> None:
    print()
    print(result.name)
    print("-" * len(result.name))
    print(f"cycles:     {result.cycles}")
    print(f"mean:       {result.mean_us:.3f} us")
    print(f"median:     {result.median_us:.3f} us")
    print(f"p95:        {result.p95_us:.3f} us")
    print(f"min:        {result.min_us:.3f} us")
    print(f"max:        {result.max_us:.3f} us")
    print(f"throughput: {result.throughput_ops:.3f} ops/s")


class _CompletionSignalingExecutor:
    """
    Wrap a transfer executor and expose an event-based completion signal.

    The wrapper intentionally does not alter the executor's behavior.
    It only records the point at which execute() returns or raises.
    """

    def __init__(self, executor) -> None:
        self._executor = executor
        self._completed = threading.Event()

    def reset(self) -> None:
        self._completed.clear()

    def execute(self, task: PrefetchTask) -> None:
        try:
            self._executor.execute(task)
        finally:
            self._completed.set()

    def wait(self, timeout: float = 5.0) -> None:
        if not self._completed.wait(timeout):
            raise AssertionError(
                f"prefetch task did not complete within {timeout}s"
            )


def _make_task(expert_id: int) -> PrefetchTask:
    return PrefetchTask(
        expert_id=expert_id,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
        priority=1.0,
        confidence=1.0,
        estimated_distance=1,
        source_path=None,
    )


def _prepare_stores(
    expert_count: int,
) -> tuple[InMemoryExpertStore, InMemoryExpertStore]:
    nvme = InMemoryExpertStore()
    ram = InMemoryExpertStore()

    for expert_id in range(expert_count):
        nvme.put(
            ExpertRecord(
                expert_id=expert_id,
                location=ExpertLocation.NVME,
                payload=f"expert-{expert_id}",
            )
        )

    return nvme, ram


def _make_engine(
    nvme: InMemoryExpertStore,
    ram: InMemoryExpertStore,
) -> tuple[PrefetchEngine, _CompletionSignalingExecutor]:
    fallback_calls: list[int] = []

    def fallback(task: PrefetchTask) -> None:
        fallback_calls.append(task.expert_id)

    transfer = create_storage_transfer_executor(
        nvme,
        ram,
        fallback,
    )

    signaling_executor = _CompletionSignalingExecutor(transfer)

    engine = PrefetchEngine(
        fallback,
        num_workers=1,
        transfer_executor=signaling_executor,
    )

    engine.start()

    return engine, signaling_executor


def _benchmark_nvme_to_ram(
    cycles: int = 1000,
) -> BenchmarkResult:
    nvme, ram = _prepare_stores(cycles)

    engine, completion = _make_engine(nvme, ram)

    samples_ns: list[int] = []

    try:
        for expert_id in range(cycles):
            task = _make_task(expert_id)

            completion.reset()

            started = time.perf_counter_ns()

            assert engine.submit(task) is True

            completion.wait()

            elapsed = (
                time.perf_counter_ns()
                - started
            )

            record = ram.get(expert_id)

            assert record is not None
            assert record.location == ExpertLocation.RAM
            assert record.payload == f"expert-{expert_id}"

            assert engine.task_status(expert_id) == PrefetchStatus.COMPLETED

            samples_ns.append(elapsed)

    finally:
        engine.stop()

    return _summarize(
        "NVMe -> RAM end-to-end",
        samples_ns,
    )


def _benchmark_submit_to_completion(
    cycles: int = 1000,
) -> BenchmarkResult:
    nvme, ram = _prepare_stores(cycles)

    engine, completion = _make_engine(nvme, ram)

    samples_ns: list[int] = []

    try:
        for expert_id in range(cycles):
            task = _make_task(expert_id)

            completion.reset()

            started = time.perf_counter_ns()

            assert engine.submit(task) is True

            completion.wait()

            samples_ns.append(
                time.perf_counter_ns() - started
            )

            assert engine.task_status(expert_id) == PrefetchStatus.COMPLETED

    finally:
        engine.stop()

    return _summarize(
        "Prefetch submit -> completion",
        samples_ns,
    )


def test_performance_baseline_nvme_to_ram() -> None:
    result = _benchmark_nvme_to_ram(
        cycles=1000,
    )

    _print_result(result)

    assert result.cycles == 1000
    assert result.throughput_ops > 0.0


def test_performance_baseline_submit_to_completion() -> None:
    result = _benchmark_submit_to_completion(
        cycles=1000,
    )

    _print_result(result)

    assert result.cycles == 1000
    assert result.throughput_ops > 0.0


def _benchmark_warm_resident(
    cycles: int = 1000,
) -> BenchmarkResult:
    nvme, ram = _prepare_stores(1)

    engine, completion = _make_engine(nvme, ram)

    try:
        # Establish RAM residency once.
        task = _make_task(0)

        completion.reset()

        assert engine.submit(task) is True

        completion.wait()

        record = ram.get(0)

        assert record is not None
        assert record.location == ExpertLocation.RAM

        samples_ns: list[int] = []

        for _ in range(cycles):
            task = _make_task(0)

            completion.reset()

            started = time.perf_counter_ns()

            assert engine.submit(task) is True

            completion.wait()

            elapsed = (
                time.perf_counter_ns()
                - started
            )

            record = ram.get(0)

            assert record is not None
            assert record.location == ExpertLocation.RAM
            assert record.payload == "expert-0"

            assert engine.task_status(0) == PrefetchStatus.COMPLETED

            samples_ns.append(elapsed)

        return _summarize(
            "Warm RAM-resident prefetch",
            samples_ns,
        )

    finally:
        engine.stop()


def test_performance_baseline_warm_resident() -> None:
    result = _benchmark_warm_resident(
        cycles=1000,
    )

    _print_result(result)

    assert result.cycles == 1000
    assert result.throughput_ops > 0.0


def _make_pipeline(
    nvme: InMemoryExpertStore,
    ram: InMemoryExpertStore,
) -> tuple[PrefetchPipeline, _CompletionSignalingExecutor]:
    from predictive_cache.cache import PredictiveExpertCache

    engine, completion = _make_engine(
        nvme,
        ram,
    )

    cache = PredictiveExpertCache()

    for expert_id in range(64):
        cache.observe([expert_id])

    scheduler = ExpertScheduler(
        cache,
        minimum_benefit=0.0,
    )

    pipeline = PrefetchPipeline(
        scheduler,
        engine,
        auto_start=False,
    )

    pipeline.start()

    return pipeline, completion


def _benchmark_pipeline_process(
    cycles: int = 1000,
) -> BenchmarkResult:
    nvme, ram = _prepare_stores(cycles)

    pipeline, completion = _make_pipeline(
        nvme,
        ram,
    )

    samples_ns: list[int] = []

    try:
        for expert_id in range(cycles):
            task = _make_task(expert_id)

            completion.reset()

            started = time.perf_counter_ns()

            accepted = pipeline.engine.submit(task)

            assert accepted is True

            completion.wait()

            elapsed = (
                time.perf_counter_ns()
                - started
            )

            record = ram.get(expert_id)

            assert record is not None
            assert record.location == ExpertLocation.RAM
            assert record.payload == f"expert-{expert_id}"

            samples_ns.append(elapsed)

    finally:
        pipeline.stop()

    return _summarize(
        "Pipeline process() dispatch latency",
        samples_ns,
    )


def test_performance_baseline_pipeline_process() -> None:
    result = _benchmark_pipeline_process(
        cycles=1000,
    )

    _print_result(result)

    assert result.cycles == 1000
    assert result.throughput_ops > 0.0


def test_performance_baseline_nvme_to_ram_10000_cycles() -> None:
    result = _benchmark_nvme_to_ram(
        cycles=10_000,
    )

    _print_result(result)

    assert result.cycles == 10_000
    assert result.throughput_ops > 0.0


def test_performance_baseline_submit_to_completion_10000_cycles() -> None:
    result = _benchmark_submit_to_completion(
        cycles=10_000,
    )

    _print_result(result)

    assert result.cycles == 10_000
    assert result.throughput_ops > 0.0


def test_performance_baseline_warm_resident_10000_cycles() -> None:
    result = _benchmark_warm_resident(
        cycles=10_000,
    )

    _print_result(result)

    assert result.cycles == 10_000
    assert result.throughput_ops > 0.0
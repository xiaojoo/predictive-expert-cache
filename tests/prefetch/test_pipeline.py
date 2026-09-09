import threading

from predictive_cache.cache import PredictiveExpertCache
from predictive_cache.scheduler import ExpertScheduler
from predictive_cache.prefetch import (
    AdmissionController,
    PrefetchEngine,
    PrefetchPipeline,
    PrefetchSource,
    PrefetchTarget,
)
from predictive_cache.scheduler import PrefetchRequest
import pytest

class FakeScheduler:
    def __init__(self, requests):
        self.requests = requests

    def plan_prefetch(self, current_experts):
        return list(self.requests)

def test_pipeline_with_real_scheduler():
    loaded = []
    loaded_event = threading.Event()

    def loader(task):
        loaded.append(task.expert_id)

        if len(loaded) == 2:
            loaded_event.set()

    cache = PredictiveExpertCache()

    # Build predictor history.
    #
    # 1 -> 2
    # 1 -> 3
    #
    # After enough observations the predictor
    # can generate prefetch candidates.
    cache.observe([1])
    cache.observe([2])
    cache.observe([1])
    cache.observe([3])

    scheduler = ExpertScheduler(cache)

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )

    pipeline = PrefetchPipeline(
        scheduler=scheduler,
        engine=engine,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
    )

    try:
        result = pipeline.process([1])

        assert result.current_experts == [1]

        assert len(result.scheduled_requests) > 0

        assert len(result.submitted_tasks) == (
            len(result.scheduled_requests)
        )

        assert all(
            task.source == PrefetchSource.NVME
            for task in result.submitted_tasks
        )

        assert all(
            task.target == PrefetchTarget.RAM
            for task in result.submitted_tasks
        )

        assert loaded_event.wait(
            timeout=2.0
        )

    finally:
        pipeline.stop()

    assert set(loaded) >= {2, 3}

def test_pipeline_uses_admission_controller():
    loaded = []

    def loader(task):
        loaded.append(task.expert_id)

    cache = PredictiveExpertCache()

    cache.observe([1])
    cache.observe([2])
    cache.observe([1])
    cache.observe([3])

    scheduler = ExpertScheduler(cache)

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )

    admission = AdmissionController(
        min_priority=0.0,
        min_confidence=1.0,
    )

    pipeline = PrefetchPipeline(
        scheduler=scheduler,
        engine=engine,
        admission=admission,
    )

    try:
        result = pipeline.process([1])

        assert len(result.scheduled_requests) > 0
        assert result.submitted_tasks == []

    finally:
        pipeline.stop()

def test_pipeline_rejects_task_by_admission():
    loaded = []

    def loader(task):
        loaded.append(task.expert_id)

    scheduler = FakeScheduler(
        [
            PrefetchRequest(
                expert_id=10,
                score=0.2,
                priority=0.5,
            )
        ]
    )

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )

    admission = AdmissionController(
        min_priority=0.0,
        min_confidence=0.8,
    )

    pipeline = PrefetchPipeline(
        scheduler=scheduler,
        engine=engine,
        admission=admission,
    )

    try:
        result = pipeline.process([1])

        assert len(result.scheduled_requests) == 1
        assert result.submitted_tasks == []
        assert loaded == []

        assert pipeline.admission_stats.total == 1
        assert pipeline.admission_stats.accepted == 0
        assert pipeline.admission_stats.rejected == 1

    finally:
        pipeline.stop()

def test_pipeline_admits_task():
    loaded = []

    def loader(task):
        loaded.append(task.expert_id)

    scheduler = FakeScheduler(
        [
            PrefetchRequest(
                expert_id=10,
                score=0.9,
                priority=0.5,
            )
        ]
    )

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )

    admission = AdmissionController(
        min_priority=0.0,
        min_confidence=0.8,
    )

    pipeline = PrefetchPipeline(
        scheduler=scheduler,
        engine=engine,
        admission=admission,
    )

    try:
        result = pipeline.process([1])

        assert len(result.scheduled_requests) == 1
        assert len(result.submitted_tasks) == 1
        assert result.submitted_tasks[0].expert_id == 10
        assert pipeline.admission_stats.total == 1
        assert pipeline.admission_stats.accepted == 1
        assert pipeline.admission_stats.rejected == 0

        assert result.admission_stats.total == 1
        assert result.admission_stats.accepted == 1
        assert result.admission_stats.rejected == 0
        assert result.admission_stats is not pipeline.admission_stats

    finally:
        pipeline.stop()

    assert loaded == [10]

def test_pipeline_propagates_admission_decisions():
    scheduler = FakeScheduler(
        [
            PrefetchRequest(
                expert_id=10,
                score=0.9,
                priority=0.9,
            ),
            PrefetchRequest(
                expert_id=11,
                score=0.2,
                priority=0.9,
            ),
            PrefetchRequest(
                expert_id=12,
                score=0.9,
                priority=0.2,
            ),
        ]
    )

    engine = PrefetchEngine(
        lambda _task: None,
        num_workers=1,
    )

    admission = AdmissionController(
        min_priority=0.5,
        min_confidence=0.8,
    )

    pipeline = PrefetchPipeline(
        scheduler=scheduler,
        engine=engine,
        admission=admission,
    )

    try:
        result = pipeline.process([1])

        assert len(result.scheduled_requests) == 3
        assert len(result.admission_decisions) == 3

        assert result.admission_decisions[0].admitted is True
        assert result.admission_decisions[0].reason.value == "accept"

        assert result.admission_decisions[1].admitted is False
        assert (
            result.admission_decisions[1].reason.value
            == "reject_low_confidence"
        )

        assert result.admission_decisions[2].admitted is False
        assert (
            result.admission_decisions[2].reason.value
            == "reject_low_priority"
        )

        assert len(result.submitted_tasks) == 1
        assert result.submitted_tasks[0].expert_id == 10

    finally:
        pipeline.stop()

def test_pipeline_result_contains_detached_admission_stats_snapshot():
    scheduler = FakeScheduler(
        [
            PrefetchRequest(
                expert_id=10,
                score=0.9,
                priority=0.9,
            )
        ]
    )

    engine = PrefetchEngine(
        lambda _task: None,
        num_workers=1,
    )

    admission = AdmissionController(
        min_priority=0.0,
        min_confidence=0.8,
    )

    pipeline = PrefetchPipeline(
        scheduler=scheduler,
        engine=engine,
        admission=admission,
    )

    try:
        # First process:
        # expert 10 is accepted.
        result1 = pipeline.process([1])

        assert result1.admission_stats is not pipeline.admission_stats

        assert result1.admission_stats.total == 1
        assert result1.admission_stats.accepted == 1
        assert result1.admission_stats.rejected == 0

        # Change the scheduled expert so the second process
        # does not trigger duplicate admission rejection.
        scheduler.requests = [
            PrefetchRequest(
                expert_id=11,
                score=0.9,
                priority=0.9,
            )
        ]

        # Second process:
        # pipeline-level stats are cumulative.
        result2 = pipeline.process([1])

        assert pipeline.admission_stats.total == 2
        assert pipeline.admission_stats.accepted == 2
        assert pipeline.admission_stats.rejected == 0

        assert result2.admission_stats.total == 2
        assert result2.admission_stats.accepted == 2
        assert result2.admission_stats.rejected == 0

        # The first result remains a snapshot of the first process.
        assert result1.admission_stats.total == 1
        assert result1.admission_stats.accepted == 1
        assert result1.admission_stats.rejected == 0

        # Each PipelineResult owns its own stats snapshot.
        assert result1.admission_stats is not result2.admission_stats

    finally:
        pipeline.stop()

def test_pipeline_propagates_scheduler_prediction_metadata():
    scheduler = FakeScheduler(
        [
            PrefetchRequest(
                expert_id=10,
                score=12.0,
                priority=6.0,
                confidence=0.8,
                estimated_distance=3.0,
            )
        ]
    )

    engine = PrefetchEngine(
        lambda _task: None,
        num_workers=1,
    )

    pipeline = PrefetchPipeline(
        scheduler=scheduler,
        engine=engine,
    )

    try:
        result = pipeline.process([1])

        assert len(result.submitted_tasks) == 1

        task = result.submitted_tasks[0]

        assert task.expert_id == 10
        assert task.confidence == pytest.approx(0.8)
        assert task.priority == pytest.approx(6.0)
        assert task.estimated_distance == 3
    finally:
        pipeline.stop()

def test_pipeline_keeps_legacy_request_compatibility():
    scheduler = FakeScheduler(
        [
            PrefetchRequest(
                expert_id=10,
                score=0.8,
                priority=0.5,
            )
        ]
    )

    engine = PrefetchEngine(
        lambda _task: None,
        num_workers=1,
    )

    pipeline = PrefetchPipeline(
        scheduler=scheduler,
        engine=engine,
    )

    try:
        result = pipeline.process([1])

        assert len(result.submitted_tasks) == 1

        task = result.submitted_tasks[0]

        assert task.confidence == pytest.approx(0.8)
        assert task.priority == pytest.approx(0.5)
        assert task.estimated_distance == 0
    finally:
        pipeline.stop()

def test_pipeline_uses_predictive_scheduler_metadata() -> None:
    cache = PredictiveExpertCache()

    cache.observe([1])
    cache.observe([2])
    cache.observe([1])
    cache.observe([3])

    scheduler = ExpertScheduler(cache)

    engine = PrefetchEngine(
        lambda _task: None,
        num_workers=1,
    )

    pipeline = PrefetchPipeline(
        scheduler=scheduler,
        engine=engine,
    )

    try:
        result = pipeline.process([1])

        assert result.submitted_tasks

        tasks = {
            task.expert_id: task
            for task in result.submitted_tasks
        }

        assert 2 in tasks
        assert 3 in tasks

        for task in tasks.values():
            assert 0.0 <= task.confidence <= 1.0
            assert task.estimated_distance >= 0
            assert task.priority >= 0.0

    finally:
        pipeline.stop()

def test_pipeline_preserves_scheduler_priority():
    class FakeScheduler:
        def plan_prefetch_predictions(self, current_experts):
            return [
                PrefetchRequest(
                    expert_id=2,
                    score=0.8,
                    priority=0.2,
                    confidence=0.8,
                    estimated_distance=4,
                ),
                PrefetchRequest(
                    expert_id=1,
                    score=0.7,
                    priority=0.9,
                    confidence=0.7,
                    estimated_distance=0,
                ),
            ]

    loaded = []

    def loader(task):
        loaded.append(task)

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )

    pipeline = PrefetchPipeline(
        scheduler=FakeScheduler(),
        engine=engine,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
    )

    try:
        result = pipeline.process([99])

        assert [task.expert_id for task in result.submitted_tasks] == [
            2,
            1,
        ]

        assert [
            task.priority
            for task in result.submitted_tasks
        ] == [
            0.2,
            0.9,
        ]

        assert [
            task.confidence
            for task in result.submitted_tasks
        ] == [
            0.8,
            0.7,
        ]

        assert [
            task.estimated_distance
            for task in result.submitted_tasks
        ] == [
            4,
            0,
        ]

    finally:
        pipeline.stop(wait=False)

def test_pipeline_task_priority_reaches_engine_queue():
    class FakeScheduler:
        def plan_prefetch_predictions(self, current_experts):
            return [
                PrefetchRequest(
                    expert_id=2,
                    score=0.8,
                    priority=0.2,
                    confidence=0.8,
                    estimated_distance=4,
                ),
                PrefetchRequest(
                    expert_id=1,
                    score=0.7,
                    priority=0.9,
                    confidence=0.7,
                    estimated_distance=0,
                ),
            ]

    def loader(task):
        pass

    engine = PrefetchEngine(
        loader,
        num_workers=1,
    )

    pipeline = PrefetchPipeline(
        scheduler=FakeScheduler(),
        engine=engine,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
    )

    try:
        result = pipeline.process([99])

        assert len(result.submitted_tasks) == 2
        assert pipeline.queue_size == 2

        first = engine._queue.get()
        second = engine._queue.get()

        assert first.expert_id == 1
        assert first.priority == 0.9

        assert second.expert_id == 2
        assert second.priority == 0.2

        engine._queue.task_done()
        engine._queue.task_done()

    finally:
        pipeline.stop(wait=False)

def test_pipeline_executes_storage_backed_nvme_to_ram():
    from predictive_cache.prefetch.storage import (
        create_storage_transfer_executor,
    )
    from predictive_cache.storage.expert_record import ExpertRecord
    from predictive_cache.storage.expert_store import ExpertLocation
    from predictive_cache.storage.nvme import NVMeExpertStore
    from predictive_cache.storage.ram import RAMExpertStore

    nvme = NVMeExpertStore()
    ram = RAMExpertStore()

    nvme.put(
        ExpertRecord(
            expert_id=61,
            location=ExpertLocation.NVME,
            payload="expert-61",
        )
    )

    def loader(task):
        raise AssertionError(
            "default loader must not handle NVMe -> RAM"
        )

    scheduler = FakeScheduler(
        [
            PrefetchRequest(
                expert_id=61,
                score=0.9,
                priority=0.9,
                confidence=0.9,
                estimated_distance=1.0,
            )
        ]
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

    pipeline = PrefetchPipeline(
        scheduler=scheduler,
        engine=engine,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
    )

    try:
        result = pipeline.process([1])

        assert len(result.scheduled_requests) == 1
        assert len(result.submitted_tasks) == 1

        task = result.submitted_tasks[0]

        assert task.expert_id == 61
        assert task.source == PrefetchSource.NVME
        assert task.target == PrefetchTarget.RAM
        assert task.priority == pytest.approx(0.9)
        assert task.confidence == pytest.approx(0.9)
        assert task.estimated_distance == 1

    finally:
        pipeline.stop()

    nvme_record = nvme.get(61)
    ram_record = ram.get(61)

    assert nvme_record is not None
    assert nvme_record.location == ExpertLocation.NVME
    assert nvme_record.payload == "expert-61"

    assert ram_record is not None
    assert ram_record.expert_id == 61
    assert ram_record.location == ExpertLocation.RAM
    assert ram_record.payload == "expert-61"

def test_pipeline_reports_storage_backed_nvme_to_ram_failure():
    from predictive_cache.prefetch.storage import (
        create_storage_transfer_executor,
    )
    from predictive_cache.storage.nvme import NVMeExpertStore
    from predictive_cache.storage.ram import RAMExpertStore
    from predictive_cache.prefetch.types import PrefetchStatus

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

    scheduler = FakeScheduler(
        [
            PrefetchRequest(
                expert_id=1000,
                score=0.9,
                priority=0.9,
            )
        ]
    )

    pipeline = PrefetchPipeline(
        scheduler,
        engine,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
    )

    pipeline.process([])

    pipeline.stop()

    results = pipeline.results()

    assert len(results) == 1

    result = results[0]

    assert result.expert_id == 1000
    assert result.status == PrefetchStatus.FAILED
    assert result.error is not None
    assert "expert 1000" in result.error

    assert nvme.get(1000) is None
    assert ram.get(1000) is None

def test_pipeline_reports_storage_backed_nvme_to_ram_success():
    from predictive_cache.prefetch.storage import (
        create_storage_transfer_executor,
    )
    from predictive_cache.prefetch.types import PrefetchStatus
    from predictive_cache.storage.expert_record import ExpertRecord
    from predictive_cache.storage.expert_store import ExpertLocation
    from predictive_cache.storage.nvme import NVMeExpertStore
    from predictive_cache.storage.ram import RAMExpertStore

    nvme = NVMeExpertStore()
    ram = RAMExpertStore()

    nvme.put(
        ExpertRecord(
            expert_id=1001,
            location=ExpertLocation.NVME,
            payload="expert-1001",
        )
    )

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

    scheduler = FakeScheduler(
        [
            PrefetchRequest(
                expert_id=1001,
                score=0.95,
                priority=0.95,
            )
        ]
    )

    pipeline = PrefetchPipeline(
        scheduler,
        engine,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
    )

    pipeline.process([])

    pipeline.stop()

    results = pipeline.results()

    assert len(results) == 1

    result = results[0]

    assert result.expert_id == 1001
    assert result.status == PrefetchStatus.COMPLETED
    assert result.error is None

    nvme_record = nvme.get(1001)
    ram_record = ram.get(1001)

    assert nvme_record is not None
    assert nvme_record.location == ExpertLocation.NVME
    assert nvme_record.payload == "expert-1001"

    assert ram_record is not None
    assert ram_record.expert_id == 1001
    assert ram_record.location == ExpertLocation.RAM
    assert ram_record.payload == "expert-1001"

def test_pipeline_executes_multiple_storage_backed_nvme_to_ram_tasks():
    from predictive_cache.prefetch.storage import (
        create_storage_transfer_executor,
    )
    from predictive_cache.prefetch.types import PrefetchStatus
    from predictive_cache.storage.expert_record import ExpertRecord
    from predictive_cache.storage.expert_store import ExpertLocation
    from predictive_cache.storage.nvme import NVMeExpertStore
    from predictive_cache.storage.ram import RAMExpertStore

    nvme = NVMeExpertStore()
    ram = RAMExpertStore()

    experts = {
        101: "expert-101",
        102: "expert-102",
        103: "expert-103",
    }

    for expert_id, payload in experts.items():
        nvme.put(
            ExpertRecord(
                expert_id=expert_id,
                location=ExpertLocation.NVME,
                payload=payload,
            )
        )

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

    scheduler = FakeScheduler(
        [
            PrefetchRequest(
                expert_id=101,
                score=0.9,
                priority=0.9,
            ),
            PrefetchRequest(
                expert_id=102,
                score=0.8,
                priority=0.8,
            ),
            PrefetchRequest(
                expert_id=103,
                score=0.7,
                priority=0.7,
            ),
        ]
    )

    pipeline = PrefetchPipeline(
        scheduler,
        engine,
        source=PrefetchSource.NVME,
        target=PrefetchTarget.RAM,
    )

    pipeline.process([])

    pipeline.stop()

    results = pipeline.results()

    assert len(results) == 3

    results_by_id = {
        result.expert_id: result
        for result in results
    }

    assert set(results_by_id) == {101, 102, 103}

    for expert_id in experts:
        result = results_by_id[expert_id]

        assert result.status == PrefetchStatus.COMPLETED
        assert result.error is None

        nvme_record = nvme.get(expert_id)
        ram_record = ram.get(expert_id)

        assert nvme_record is not None
        assert nvme_record.location == ExpertLocation.NVME
        assert nvme_record.payload == experts[expert_id]

        assert ram_record is not None
        assert ram_record.expert_id == expert_id
        assert ram_record.location == ExpertLocation.RAM
        assert ram_record.payload == experts[expert_id]
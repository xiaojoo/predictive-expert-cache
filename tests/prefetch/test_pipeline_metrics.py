from predictive_cache.prefetch.admission import AdmissionDecision, AdmissionReason
from predictive_cache.prefetch.admission_stats import AdmissionStats
from predictive_cache.prefetch.pipeline import PrefetchPipeline


class FakeScheduler:
    def plan_prefetch(self, current_experts):
        return []


class FakeEngine:
    running = False
    queue_size = 0

    def start(self):
        pass

    def stop(self, *, wait=True):
        pass

    def results(self):
        return []

    def clear_results(self):
        pass


def test_pipeline_metrics_returns_admission_metrics():
    stats = AdmissionStats(
        total=10,
        accepted=6,
        rejected=4,
        reject_low_priority=1,
        reject_low_confidence=1,
        reject_duplicate_queued=1,
        reject_duplicate_running=0,
        reject_queue_full=1,
    )

    pipeline = PrefetchPipeline(
        scheduler=FakeScheduler(),
        engine=FakeEngine(),
        auto_start=False,
        admission_stats=stats,
    )

    metrics = pipeline.metrics()

    assert metrics["admission_total"] == 10
    assert metrics["admission_accepted"] == 6
    assert metrics["admission_rejected"] == 4

    assert metrics["admission_reject_low_priority"] == 1
    assert metrics["admission_reject_low_confidence"] == 1
    assert metrics["admission_reject_duplicate_queued"] == 1
    assert metrics["admission_reject_duplicate_running"] == 0
    assert metrics["admission_reject_queue_full"] == 1

    assert metrics["admission_acceptance_rate"] == 0.6
    assert metrics["admission_rejection_rate"] == 0.4


def test_pipeline_metrics_is_detached_from_internal_stats():
    stats = AdmissionStats(
        total=5,
        accepted=3,
        rejected=2,
    )

    pipeline = PrefetchPipeline(
        scheduler=FakeScheduler(),
        engine=FakeEngine(),
        auto_start=False,
        admission_stats=stats,
    )

    metrics = pipeline.metrics()

    metrics["admission_total"] = 999
    metrics["admission_accepted"] = 999

    assert pipeline.admission_stats.total == 5
    assert pipeline.admission_stats.accepted == 3


def test_pipeline_metrics_returns_fresh_snapshot():
    stats = AdmissionStats(
        total=2,
        accepted=1,
        rejected=1,
    )

    pipeline = PrefetchPipeline(
        scheduler=FakeScheduler(),
        engine=FakeEngine(),
        auto_start=False,
        admission_stats=stats,
    )

    metrics1 = pipeline.metrics()
    metrics2 = pipeline.metrics()

    assert metrics1 == metrics2
    assert metrics1 is not metrics2

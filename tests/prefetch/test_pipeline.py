import threading

from predictive_cache.cache import PredictiveExpertCache
from predictive_cache.scheduler import ExpertScheduler
from predictive_cache.prefetch import (
    PrefetchEngine,
    PrefetchPipeline,
    PrefetchSource,
    PrefetchTarget,
)


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
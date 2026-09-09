from __future__ import annotations

from predictive_cache.cache import PredictiveExpertCache
from predictive_cache.scheduler import ExpertScheduler
from predictive_cache.types import CacheConfig, ExpertPrediction


def test_explicit_capacity_state_prevents_overcommit() -> None:
    cache = PredictiveExpertCache(
        CacheConfig(
            capacity=4,
            prediction_top_k=1,
        )
    )

    cache.insert(0, size_bytes=1024 * 1024)
    cache.insert(1, size_bytes=512 * 1024)

    assert cache.size == 2

    actual_used_mb = sum(
        entry.size_bytes / (1024 * 1024)
        for expert_id in cache.cached_experts()
        if (entry := cache.get(expert_id)) is not None
    )

    scheduler = ExpertScheduler(
        cache,
        cache_capacity_mb=2.0,
        cache_used_mb=actual_used_mb,
        default_expert_size_mb=1.0,
    )

    predictions = [
        ExpertPrediction(
            expert_id=2,
            score=0.9,
        ),
    ]

    requests = scheduler.plan_predictions(
        predictions,
        expert_sizes_mb={2: 1.0},
    )

    print()
    print("7-H-4-3 Explicit Capacity State")
    print("--------------------------------")
    print(f"cache residents:     {cache.cached_experts()}")
    print(f"actual used MB:      {actual_used_mb:.2f}")
    print("scheduler capacity:  2.00 MB")
    print(f"scheduler used_mb:   {actual_used_mb:.2f} MB")
    print("candidate size:      1.00 MB")
    print(f"scheduled requests:  {len(requests)}")

    # 1.50 MB is already used, leaving only 0.50 MB.
    # A 1.00 MB expert cannot fit.
    assert requests == []

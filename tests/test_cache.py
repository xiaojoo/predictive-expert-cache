from predictive_cache import (
    CacheConfig,
    PredictiveExpertCache,
)


def test_cache_insert():

    cache = PredictiveExpertCache(
        CacheConfig(capacity=2)
    )

    cache.insert(1)
    cache.insert(2)

    assert cache.cached_experts() == [1, 2]


def test_cache_eviction():

    cache = PredictiveExpertCache(
        CacheConfig(capacity=2)
    )

    cache.insert(1)
    cache.insert(2)

    evicted = cache.insert(3)

    assert evicted == 1

    assert cache.cached_experts() == [2, 3]


def test_cache_hit():

    cache = PredictiveExpertCache(
        CacheConfig(capacity=2)
    )

    cache.insert(1)

    entry = cache.get(1)

    assert entry is not None
    assert entry.expert_id == 1
    assert entry.hit_count >= 1


def test_cache_miss():

    cache = PredictiveExpertCache(
        CacheConfig(capacity=2)
    )

    assert cache.get(999) is None


def test_prefetch_candidates():

    cache = PredictiveExpertCache(
        CacheConfig(
            capacity=4,
            prediction_top_k=3,
        )
    )

    for _ in range(10):
        cache.observe([1])

    for _ in range(5):
        cache.observe([2])

    cache.observe([1])
    cache.observe([3])

    predictions = cache.prefetch_candidates(
        [1]
    )

    assert predictions
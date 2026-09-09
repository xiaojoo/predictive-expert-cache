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

def test_observe_does_not_create_cache_hits():
    cache = PredictiveExpertCache(
        CacheConfig(capacity=2)
    )

    cache.insert(1)

    cache.observe([1])

    entry = cache.lru.peek(1)

    assert entry is not None
    assert entry.hit_count == 0

    assert cache.predictor.stats[1].frequency == 1
    assert cache.predictor.stats[1].hit_count == 0


def test_cache_lookup_updates_predictor_hit_and_miss_stats():
    cache = PredictiveExpertCache(
        CacheConfig(capacity=2)
    )

    cache.insert(1)

    assert cache.get(1) is not None
    assert cache.get(999) is None

    assert cache.predictor.stats[1].hit_count == 1
    assert cache.predictor.stats[999].miss_count == 1


def test_insert_rejects_negative_size():
    cache = PredictiveExpertCache()

    try:
        cache.insert(
            1,
            size_bytes=-1,
        )
    except ValueError as exc:
        assert str(exc) == (
            "size_bytes must be >= 0"
        )
    else:
        raise AssertionError(
            "negative size_bytes must be rejected"
        )
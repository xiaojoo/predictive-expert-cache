from predictive_cache.lru import ExpertLRU


def test_lru_eviction():

    cache = ExpertLRU[int, str](capacity=2)

    cache.put(1, "expert-1")
    cache.put(2, "expert-2")

    assert cache.keys() == [1, 2]

    cache.get(1)

    assert cache.keys() == [2, 1]

    evicted = cache.put(3, "expert-3")

    assert evicted == 2

    assert cache.keys() == [1, 3]


def test_lru_update_existing():

    cache = ExpertLRU[int, str](capacity=2)

    cache.put(1, "a")
    cache.put(2, "b")

    cache.put(1, "updated")

    assert cache.get(1) == "updated"
    assert cache.keys() == [2, 1]


def test_lru_remove():

    cache = ExpertLRU[int, str](capacity=2)

    cache.put(1, "a")
    cache.put(2, "b")

    assert cache.remove(1) == "a"

    assert cache.keys() == [2]
import pytest

from predictive_cache.prefetch import InMemoryPrefetchRam


def test_ram_stores_and_gets_data():
    ram = InMemoryPrefetchRam()

    ram.store(1, "expert-1")

    assert ram.get(1) == "expert-1"
    assert ram.contains(1) is True


def test_ram_returns_none_for_missing_expert():
    ram = InMemoryPrefetchRam()

    assert ram.get(999) is None
    assert ram.contains(999) is False


def test_ram_replaces_existing_expert():
    ram = InMemoryPrefetchRam()

    ram.store(1, "old")
    ram.store(1, "new")

    assert ram.get(1) == "new"


def test_ram_clear():
    ram = InMemoryPrefetchRam()

    ram.store(1, "expert-1")
    ram.store(2, "expert-2")

    ram.clear()

    assert ram.get(1) is None
    assert ram.get(2) is None
    assert ram.contains(1) is False
    assert ram.contains(2) is False


def test_ram_rejects_negative_expert_id():
    ram = InMemoryPrefetchRam()

    with pytest.raises(ValueError, match="expert_id must be >= 0"):
        ram.store(-1, "invalid")
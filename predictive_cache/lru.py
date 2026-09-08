from __future__ import annotations

from collections import OrderedDict
from typing import Generic, Iterator, TypeVar


K = TypeVar("K")
V = TypeVar("V")


class ExpertLRU(Generic[K, V]):
    """
    Simple O(1) LRU cache.

    The class only manages cache ordering.
    It does not know anything about GPU/RAM/NVMe.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")

        self.capacity = capacity
        self._items: OrderedDict[K, V] = OrderedDict()

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key: K) -> bool:
        return key in self._items

    def __iter__(self) -> Iterator[K]:
        return iter(self._items)

    def get(self, key: K, default: V | None = None) -> V | None:
        if key not in self._items:
            return default

        value = self._items.pop(key)
        self._items[key] = value

        return value

    def peek(self, key: K, default: V | None = None) -> V | None:
        return self._items.get(key, default)

    def put(self, key: K, value: V) -> K | None:
        evicted = None

        if key in self._items:
            self._items.pop(key)

        self._items[key] = value

        if len(self._items) > self.capacity:
            evicted, _ = self._items.popitem(last=False)

        return evicted

    def remove(self, key: K) -> V | None:
        return self._items.pop(key, None)

    def clear(self) -> None:
        self._items.clear()

    def keys(self) -> list[K]:
        return list(self._items.keys())

    def values(self) -> list[V]:
        return list(self._items.values())

    def items(self) -> list[tuple[K, V]]:
        return list(self._items.items())

    @property
    def oldest_key(self) -> K | None:
        if not self._items:
            return None

        return next(iter(self._items))

    @property
    def newest_key(self) -> K | None:
        if not self._items:
            return None

        return next(reversed(self._items))
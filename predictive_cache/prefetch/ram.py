from __future__ import annotations


class InMemoryPrefetchRam:
    """Minimal in-memory RAM target for prefetch transfers."""

    def __init__(self) -> None:
        self._data: dict[int, object] = {}

    def store(self, expert_id: int, data: object) -> None:
        if expert_id < 0:
            raise ValueError("expert_id must be >= 0")

        self._data[expert_id] = data

    def get(self, expert_id: int) -> object | None:
        return self._data.get(expert_id)

    def contains(self, expert_id: int) -> bool:
        return expert_id in self._data

    def clear(self) -> None:
        self._data.clear()
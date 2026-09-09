from predictive_cache.storage.expert_record import ExpertRecord
from predictive_cache.storage.expert_store import ExpertLocation, ExpertStore


class RAMExpertStore(ExpertStore):
    """Minimal RAM storage backend."""

    def __init__(self) -> None:
        self._records: dict[str, ExpertRecord] = {}

    def put(self, record: ExpertRecord) -> None:
        self._records[record.expert_id] = record

    def get(self, expert_id: str) -> ExpertRecord | None:
        return self._records.get(expert_id)

    def contains(self, expert_id: str) -> bool:
        return expert_id in self._records

    def remove(self, expert_id: str) -> None:
        self._records.pop(expert_id, None)

    def load(self, expert_id: str) -> None:
        raise NotImplementedError("RAM backend does not load into GPU directly")

    def unload(self, expert_id: str) -> None:
        raise NotImplementedError("RAM backend does not unload to another backend")

    def prefetch(self, expert_id: str) -> None:
        raise NotImplementedError("RAM backend prefetch is not implemented yet")

    def location(self, expert_id: str) -> ExpertLocation:
        return ExpertLocation.RAM
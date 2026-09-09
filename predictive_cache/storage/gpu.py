from predictive_cache.storage.expert_record import ExpertRecord
from predictive_cache.storage.expert_store import ExpertLocation, ExpertStore


class GPUExpertStore(ExpertStore):
    """Minimal GPU storage backend."""

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
        raise NotImplementedError("GPU backend load is implemented in the CUDA phase")

    def unload(self, expert_id: str) -> None:
        raise NotImplementedError("GPU backend unload is implemented in the CUDA phase")

    def prefetch(self, expert_id: str) -> None:
        raise NotImplementedError("GPU backend prefetch is implemented in the CUDA phase")

    def location(self, expert_id: str) -> ExpertLocation:
        return ExpertLocation.GPU
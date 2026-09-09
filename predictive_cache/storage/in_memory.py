from predictive_cache.storage.expert_store import ExpertLocation, ExpertStore
from predictive_cache.storage.expert_record import ExpertRecord


class InMemoryExpertStore(ExpertStore):
    """In-memory logical expert store used for storage lifecycle tests."""

    _ORDER = (
        ExpertLocation.REMOTE,
        ExpertLocation.NVME,
        ExpertLocation.RAM,
        ExpertLocation.GPU,
    )

    def __init__(self) -> None:
        self._locations: dict[str, ExpertLocation] = {}
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
        current = self.location(expert_id)

        if current == ExpertLocation.RAM:
            self._locations[expert_id] = ExpertLocation.GPU
            return

        if current == ExpertLocation.GPU:
            return

        raise ValueError(
            f"expert {expert_id!r} must be in RAM before load; "
            f"current location: {current.value}"
        )

    def unload(self, expert_id: str) -> None:
        current = self.location(expert_id)

        if current == ExpertLocation.GPU:
            self._locations[expert_id] = ExpertLocation.RAM
            return

        if current == ExpertLocation.RAM:
            self._locations[expert_id] = ExpertLocation.NVME
            return

        if current == ExpertLocation.NVME:
            self._locations[expert_id] = ExpertLocation.REMOTE
            return

        if current == ExpertLocation.REMOTE:
            return

        raise AssertionError(f"unknown expert location: {current!r}")

    def prefetch(self, expert_id: str) -> None:
        current = self.location(expert_id)

        if current == ExpertLocation.REMOTE:
            self._locations[expert_id] = ExpertLocation.NVME
            return

        if current == ExpertLocation.NVME:
            self._locations[expert_id] = ExpertLocation.RAM
            return

        if current in (ExpertLocation.RAM, ExpertLocation.GPU):
            return

        raise AssertionError(f"unknown expert location: {current!r}")

    def location(self, expert_id: str) -> ExpertLocation:
        return self._locations.get(expert_id, ExpertLocation.REMOTE)

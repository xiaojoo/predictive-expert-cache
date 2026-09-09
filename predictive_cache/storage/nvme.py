from __future__ import annotations

import pickle
from pathlib import Path

from predictive_cache.storage.expert_record import ExpertRecord
from predictive_cache.storage.expert_store import ExpertLocation, ExpertStore


class NVMeExpertStore(ExpertStore):
    """NVMe-backed expert storage.

    Without root_dir this preserves the original in-memory compatibility
    behavior. With root_dir, each expert is persisted as an individual file
    on the filesystem.
    """

    def __init__(self, root_dir: str | Path | None = None) -> None:
        self._root_dir = (
            Path(root_dir).resolve()
            if root_dir is not None
            else None
        )

        self._records: dict[str, ExpertRecord] = {}

        if self._root_dir is not None:
            self._root_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

    @property
    def root_dir(self) -> Path | None:
        return self._root_dir

    def _path(self, expert_id: str) -> Path:
        if self._root_dir is None:
            raise RuntimeError(
                "filesystem path is unavailable for in-memory NVMe store"
            )

        safe_id = str(expert_id).replace("/", "_").replace("\\", "_")
        return self._root_dir / f"{safe_id}.pkl"

    def put(self, record: ExpertRecord) -> None:
        if self._root_dir is None:
            self._records[record.expert_id] = record
            return

        path = self._path(record.expert_id)

        with path.open("wb") as handle:
            pickle.dump(
                record,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def get(self, expert_id: str) -> ExpertRecord | None:
        if self._root_dir is None:
            return self._records.get(expert_id)

        path = self._path(expert_id)

        if not path.exists():
            return None

        with path.open("rb") as handle:
            return pickle.load(handle)

    def contains(self, expert_id: str) -> bool:
        if self._root_dir is None:
            return expert_id in self._records

        return self._path(expert_id).exists()

    def remove(self, expert_id: str) -> None:
        if self._root_dir is None:
            self._records.pop(expert_id, None)
            return

        self._path(expert_id).unlink(missing_ok=True)

    def load(self, expert_id: str) -> None:
        raise NotImplementedError(
            "NVMe backend does not load into GPU directly"
        )

    def unload(self, expert_id: str) -> None:
        raise NotImplementedError(
            "NVMe backend does not unload yet"
        )

    def prefetch(self, expert_id: str) -> None:
        raise NotImplementedError(
            "NVMe backend prefetch is not implemented yet"
        )

    def location(self, expert_id: str) -> ExpertLocation:
        return ExpertLocation.NVME

from __future__ import annotations

import torch

from predictive_cache.storage.expert_record import ExpertRecord
from predictive_cache.storage.expert_store import ExpertLocation, ExpertStore
from predictive_cache.types import ExpertId


class GPUExpertStore(ExpertStore):
    """
    GPU storage backend.

    Tensor payloads are moved between CPU and CUDA.
    Non-tensor payloads are kept unchanged for compatibility.
    """

    def __init__(self, device: str | torch.device = "cuda") -> None:
        if isinstance(device, str) and device == "cuda":
            self._device = torch.device(
                "cuda",
                torch.cuda.current_device(),
            )
        elif isinstance(device, torch.device) and device.type == "cuda":
            index = (
                torch.cuda.current_device()
                if device.index is None
                else device.index
            )
            self._device = torch.device("cuda", index)
        else:
            self._device = torch.device(device)

        self._records: dict[ExpertId, ExpertRecord] = {}

    @property
    def device(self) -> torch.device:
        """Return the configured device."""
        return self._device

    def _require_cuda(self) -> None:
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

    def _to_gpu(self, payload: object) -> object:
        self._require_cuda()

        if isinstance(payload, torch.Tensor):
            return payload.to(self._device)

        return payload

    def _to_cpu(self, payload: object) -> object:
        if isinstance(payload, torch.Tensor):
            return payload.to("cpu")

        return payload

    def put(self, record: ExpertRecord) -> None:
        """Store an expert on the configured GPU."""
        record.payload = self._to_gpu(record.payload)
        record.location = ExpertLocation.GPU
        self._records[record.expert_id] = record

    def get(self, expert_id: ExpertId) -> ExpertRecord | None:
        return self._records.get(expert_id)

    def contains(self, expert_id: ExpertId) -> bool:
        return expert_id in self._records

    def remove(self, expert_id: ExpertId) -> None:
        self._records.pop(expert_id, None)

    def load(self, expert_id: ExpertId) -> None:
        """Move an existing expert onto the configured GPU."""
        self._require_cuda()

        record = self._records.get(expert_id)

        if record is None:
            raise KeyError(f"expert {expert_id!r} not found")

        record.payload = self._to_gpu(record.payload)
        record.location = ExpertLocation.GPU

    def unload(self, expert_id: ExpertId) -> None:
        """Move an existing expert back to CPU memory."""
        record = self._records.get(expert_id)

        if record is None:
            raise KeyError(f"expert {expert_id!r} not found")

        record.payload = self._to_cpu(record.payload)
        record.location = ExpertLocation.RAM

    def prefetch(self, expert_id: ExpertId) -> None:
        """Prepare an expert for GPU execution."""
        self.load(expert_id)

    def location(self, expert_id: ExpertId) -> ExpertLocation:
        record = self._records.get(expert_id)

        if record is None:
            return ExpertLocation.GPU

        return record.location

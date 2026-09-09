from abc import ABC, abstractmethod
from enum import Enum

from predictive_cache.types import ExpertId


class ExpertLocation(str, Enum):
    """Physical/logical location of an expert."""

    GPU = "gpu"
    RAM = "ram"
    NVME = "nvme"
    REMOTE = "remote"


class ExpertStore(ABC):
    """Abstract storage interface for expert lifecycle operations."""

    @abstractmethod
    def load(self, expert_id: ExpertId) -> None:
        """Load an expert into the active execution location."""
        raise NotImplementedError

    @abstractmethod
    def unload(self, expert_id: ExpertId) -> None:
        """Unload an expert from the active execution location."""
        raise NotImplementedError

    @abstractmethod
    def prefetch(self, expert_id: ExpertId) -> None:
        """Prepare an expert for a future load."""
        raise NotImplementedError

    @abstractmethod
    def location(self, expert_id: ExpertId) -> ExpertLocation | None:
        """Return the current storage location of an expert."""
        raise NotImplementedError

    @abstractmethod
    def put(self, record: "ExpertRecord") -> None:
        """Store an expert record."""
        raise NotImplementedError

    @abstractmethod
    def get(self, expert_id: ExpertId) -> "ExpertRecord | None":
        """Return an expert record, or None when absent."""
        raise NotImplementedError

    @abstractmethod
    def contains(self, expert_id: ExpertId) -> bool:
        """Return whether an expert record exists."""
        raise NotImplementedError

    @abstractmethod
    def remove(self, expert_id: ExpertId) -> None:
        """Remove an expert record if present."""
        raise NotImplementedError
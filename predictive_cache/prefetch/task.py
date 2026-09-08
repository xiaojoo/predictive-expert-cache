from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TaskState(str, Enum):
    """State of a prefetch task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class PrefetchTask:
    """A single expert prefetch task."""

    expert_id: int
    priority: float
    probability: float

    source: str = "prediction"

    def __post_init__(self) -> None:
        if self.expert_id < 0:
            raise ValueError("expert_id must be non-negative")

        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be between 0 and 1")

        if self.priority < 0.0:
            raise ValueError("priority must be non-negative")


@dataclass
class TaskResult:
    """Execution result of a prefetch task."""

    task: PrefetchTask
    state: TaskState

    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.state == TaskState.COMPLETED
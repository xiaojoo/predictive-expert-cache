from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PrefetchSource(str, Enum):
    NVME = "nvme"
    RAM = "ram"


class PrefetchTarget(str, Enum):
    RAM = "ram"
    GPU = "gpu"


class PrefetchStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class PrefetchTask:
    expert_id: int

    source: PrefetchSource
    target: PrefetchTarget

    priority: float = 0.0

    # Predictor/Scheduler 可以把预测置信度传进来
    confidence: float = 0.0

    # 当前 expert 预计多久以后需要
    estimated_distance: int = 0

    # 可选路径，例如：
    # H:/experts/expert_41.safetensors
    source_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.expert_id < 0:
            raise ValueError("expert_id must be >= 0")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

        if self.estimated_distance < 0:
            raise ValueError("estimated_distance must be >= 0")


@dataclass
class PrefetchResult:
    expert_id: int
    status: PrefetchStatus

    source: PrefetchSource
    target: PrefetchTarget

    priority: float

    error: Optional[str] = None
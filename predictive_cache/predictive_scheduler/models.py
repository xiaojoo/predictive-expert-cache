from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SchedulerAction(str, Enum):
    """Action selected by the predictive predictive_scheduler."""

    PREFETCH = "prefetch"
    WAIT = "wait"
    EVICT = "evict"


@dataclass(frozen=True)
class ExpertPrediction:
    """
    Prediction result for one expert.

    probability:
        Predicted probability that this expert will be needed soon.

    expert_size_mb:
        Approximate memory footprint of the expert.

    """

    expert_id: int
    probability: float
    expert_size_mb: float

    def __post_init__(self) -> None:
        if self.expert_id < 0:
            raise ValueError("expert_id must be >= 0")

        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be between 0 and 1")

        if self.expert_size_mb <= 0:
            raise ValueError("expert_size_mb must be > 0")


@dataclass(frozen=True)
class TransferCost:
    """
    Estimated transfer costs for one expert.

    gpu_to_ram_ms:
        Time required to move the expert from GPU memory to RAM.

    ram_to_gpu_ms:
        Time required to move the expert from RAM to GPU memory.
    """

    gpu_to_ram_ms: float
    ram_to_gpu_ms: float

    def __post_init__(self) -> None:
        if self.gpu_to_ram_ms < 0:
            raise ValueError("gpu_to_ram_ms must be >= 0")

        if self.ram_to_gpu_ms < 0:
            raise ValueError("ram_to_gpu_ms must be >= 0")


@dataclass(frozen=True)
class ExpertState:
    """
    Current location/state of an expert.
    """

    expert_id: int
    size_mb: float
    on_gpu: bool = False
    on_ram: bool = False

    def __post_init__(self) -> None:
        if self.expert_id < 0:
            raise ValueError("expert_id must be >= 0")

        if self.size_mb <= 0:
            raise ValueError("size_mb must be > 0")

        if self.on_gpu and self.on_ram:
            raise ValueError(
                "An expert cannot be marked as both on_gpu and on_ram"
            )


@dataclass(frozen=True)
class CacheState:
    """
    Current GPU cache capacity.
    """

    capacity_mb: float
    used_mb: float

    def __post_init__(self) -> None:
        if self.capacity_mb <= 0:
            raise ValueError("capacity_mb must be > 0")

        if self.used_mb < 0:
            raise ValueError("used_mb must be >= 0")

        if self.used_mb > self.capacity_mb:
            raise ValueError(
                "used_mb cannot exceed capacity_mb"
            )

    @property
    def free_mb(self) -> float:
        return self.capacity_mb - self.used_mb


@dataclass(frozen=True)
class SchedulerInput:
    """
    Complete input required by the predictive predictive_scheduler.

    This object deliberately contains no CUDA/PyTorch dependency.
    """

    prediction: ExpertPrediction
    transfer_cost: TransferCost
    cache_state: CacheState
    expert_state: ExpertState

    # Cost of reusing an already resident expert.
    reuse_cost: float = 1.0

    # Cost associated with evicting an expert.
    eviction_cost: float = 0.0

    # Optional system bandwidth information.
    bandwidth_mb_s: float | None = None

    def __post_init__(self) -> None:
        if self.reuse_cost < 0:
            raise ValueError("reuse_cost must be >= 0")

        if self.eviction_cost < 0:
            raise ValueError("eviction_cost must be >= 0")

        if self.bandwidth_mb_s is not None and self.bandwidth_mb_s <= 0:
            raise ValueError("bandwidth_mb_s must be > 0")


@dataclass(frozen=True)
class SchedulerDecision:
    """
    Decision produced by the predictive predictive_scheduler.
    """

    expert_id: int
    action: SchedulerAction

    score: float = 0.0
    reuse_value: float = 0.0
    transfer_cost: float = 0.0
    eviction_cost: float = 0.0

    reason: str = ""

    @property
    def should_prefetch(self) -> bool:
        return self.action is SchedulerAction.PREFETCH

    @property
    def should_wait(self) -> bool:
        return self.action is SchedulerAction.WAIT

    @property
    def should_evict(self) -> bool:
        return self.action is SchedulerAction.EVICT
from __future__ import annotations

from dataclasses import dataclass

from .models import ExpertState


@dataclass(frozen=True)
class EvictionCandidate:
    """
    可被驱逐的 Expert。

    eviction_priority 越小，越应该优先驱逐。
    """

    expert_id: int
    size_mb: float
    eviction_priority: float

    def __post_init__(self) -> None:
        if self.expert_id < 0:
            raise ValueError("expert_id must be >= 0")

        if self.size_mb <= 0:
            raise ValueError("size_mb must be > 0")


@dataclass(frozen=True)
class CapacityManager:
    """
    GPU Expert Cache 容量管理器。

    当前阶段只负责：

    1. 判断 Expert 是否能够放入 GPU Cache
    2. 计算需要释放多少空间
    3. 选择需要 Evict 的 Expert

    不负责：

    - Expert 预测
    - PREFETCH / WAIT / EVICT 最终策略
    - 真正的数据搬运
    """

    capacity_mb: float
    used_mb: float = 0.0

    def __post_init__(self) -> None:
        if self.capacity_mb <= 0:
            raise ValueError("capacity_mb must be > 0")

        if self.used_mb < 0:
            raise ValueError("used_mb must be >= 0")

        if self.used_mb > self.capacity_mb:
            raise ValueError("used_mb cannot exceed capacity_mb")

    @property
    def free_mb(self) -> float:
        """GPU Cache 当前剩余容量。"""
        return self.capacity_mb - self.used_mb

    def can_fit(self, expert_size_mb: float) -> bool:
        """
        判断指定 Expert 是否可以直接放入当前 Cache。
        """
        if expert_size_mb <= 0:
            raise ValueError("expert_size_mb must be > 0")

        return expert_size_mb <= self.free_mb

    def required_free_mb(self, expert_size_mb: float) -> float:
        """
        计算为了放入 Expert 需要额外释放多少空间。

        如果当前空间足够，则返回 0。
        """
        if expert_size_mb <= 0:
            raise ValueError("expert_size_mb must be > 0")

        return max(0.0, expert_size_mb - self.free_mb)

    def reserve(self, expert_size_mb: float) -> "CapacityManager":
        """
        模拟预留 GPU Cache 空间。

        如果空间不足，则抛出 ValueError。

        注意：
        这里只更新容量状态，不执行真正的 Expert 加载。
        """
        if expert_size_mb <= 0:
            raise ValueError("expert_size_mb must be > 0")

        if not self.can_fit(expert_size_mb):
            raise ValueError(
                f"insufficient cache capacity: "
                f"required={expert_size_mb:.2f} MB, "
                f"free={self.free_mb:.2f} MB"
            )

        return CapacityManager(
            capacity_mb=self.capacity_mb,
            used_mb=self.used_mb + expert_size_mb,
        )

    def release(self, expert_size_mb: float) -> "CapacityManager":
        """
        模拟释放 GPU Cache 空间。
        """
        if expert_size_mb <= 0:
            raise ValueError("expert_size_mb must be > 0")

        if expert_size_mb > self.used_mb:
            raise ValueError(
                f"cannot release {expert_size_mb:.2f} MB: "
                f"only {self.used_mb:.2f} MB is currently used"
            )

        return CapacityManager(
            capacity_mb=self.capacity_mb,
            used_mb=self.used_mb - expert_size_mb,
        )

    @staticmethod
    def select_eviction_candidates(
        candidates: list[EvictionCandidate],
        required_free_mb: float,
    ) -> list[EvictionCandidate]:
        """
        从候选 Expert 中选择足够释放空间的 Expert。

        选择规则：

        1. eviction_priority 从小到大
        2. 如果 priority 相同，则优先选择更大的 Expert

        这样可以尽量用较少的 Eviction 操作释放足够空间。
        """
        if required_free_mb < 0:
            raise ValueError("required_free_mb must be >= 0")

        if required_free_mb == 0:
            return []

        ordered = sorted(
            candidates,
            key=lambda item: (
                item.eviction_priority,
                -item.size_mb,
            ),
        )

        selected: list[EvictionCandidate] = []
        released_mb = 0.0

        for candidate in ordered:
            selected.append(candidate)
            released_mb += candidate.size_mb

            if released_mb >= required_free_mb:
                break

        if released_mb < required_free_mb:
            raise ValueError(
                f"insufficient eviction candidates: "
                f"required={required_free_mb:.2f} MB, "
                f"available={released_mb:.2f} MB"
            )

        return selected

    @staticmethod
    def candidates_from_states(
        states: list[ExpertState],
        priorities: dict[int, float] | None = None,
    ) -> list[EvictionCandidate]:
        """
        将 ExpertState 转换为 EvictionCandidate。

        只有当前位于 GPU 的 Expert 才能作为 Eviction Candidate。

        priorities:
            expert_id -> eviction priority

        未提供 priority 时默认使用 0。
        """
        priorities = priorities or {}

        candidates: list[EvictionCandidate] = []

        for state in states:
            if not state.on_gpu:
                continue

            candidates.append(
                EvictionCandidate(
                    expert_id=state.expert_id,
                    size_mb=state.size_mb,
                    eviction_priority=priorities.get(
                        state.expert_id,
                        0.0,
                    ),
                )
            )

        return candidates
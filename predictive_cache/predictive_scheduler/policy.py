from __future__ import annotations

from dataclasses import dataclass

from .capacity import CapacityManager, EvictionCandidate
from .cost_model import CostModel
from .models import (
    ExpertPrediction,
    ExpertState,
    SchedulerAction,
    SchedulerDecision,
    TransferCost,
)


@dataclass(frozen=True)
class SchedulerPolicy:
    """
    Predictive Expert Cache 调度策略。

    Policy 负责决定：

        PREFETCH
        WAIT
        EVICT

    不负责真正执行 Expert 的加载、卸载和数据搬运。
    """

    cost_model: CostModel
    minimum_benefit: float = 0.0

    def __post_init__(self) -> None:
        if self.minimum_benefit < 0:
            raise ValueError("minimum_benefit must be >= 0")

    def decide(
        self,
        prediction: ExpertPrediction,
        transfer_cost: TransferCost,
        cache: CapacityManager,
        *,
        reuse_cost_ms: float | None = None,
        eviction_cost_ms: float = 0.0,
        bandwidth_mb_s: float | None = None,
    ) -> SchedulerDecision:
        """
        对单个 Expert 做调度决策。

        决策流程：

        1. 已经在 GPU → WAIT
        2. 计算 benefit
        3. benefit 不足 → WAIT
        4. GPU 有空间 → PREFETCH
        5. GPU 空间不足 → EVICT
        """

        benefit = self.cost_model.benefit(
            prediction,
            transfer_cost,
            reuse_cost_ms=reuse_cost_ms,
            eviction_cost_ms=eviction_cost_ms,
            bandwidth_mb_s=bandwidth_mb_s,
        )

        reuse_value = self.cost_model.reuse_value(
            prediction,
            reuse_cost_ms,
        )

        transfer = self.cost_model.transfer_cost(
            prediction,
            transfer_cost,
            bandwidth_mb_s=bandwidth_mb_s,
        )

        # if benefit < self.minimum_benefit:
        if benefit <= self.minimum_benefit:
            return SchedulerDecision(
                expert_id=prediction.expert_id,
                action=SchedulerAction.WAIT,
                score=benefit,
                reuse_value=reuse_value,
                transfer_cost=transfer,
                eviction_cost=eviction_cost_ms,
                reason=(
                    "benefit is below the prefetch threshold"
                ),
            )

        if cache.can_fit(prediction.expert_size_mb):
            return SchedulerDecision(
                expert_id=prediction.expert_id,
                action=SchedulerAction.PREFETCH,
                score=benefit,
                reuse_value=reuse_value,
                transfer_cost=transfer,
                eviction_cost=eviction_cost_ms,
                reason=(
                    "positive benefit and sufficient GPU cache capacity"
                ),
            )

        return SchedulerDecision(
            expert_id=prediction.expert_id,
            action=SchedulerAction.EVICT,
            score=benefit,
            reuse_value=reuse_value,
            transfer_cost=transfer,
            eviction_cost=eviction_cost_ms,
            reason=(
                "positive benefit but insufficient GPU cache capacity; "
                "eviction is required"
            ),
        )

    def select_evictions(
        self,
        prediction: ExpertPrediction,
        cache: CapacityManager,
        candidates: list[EvictionCandidate],
    ) -> list[EvictionCandidate]:
        """
        为目标 Expert 选择需要驱逐的 GPU Experts。
        """

        required_free = cache.required_free_mb(
            prediction.expert_size_mb
        )

        if required_free == 0:
            return []

        return CapacityManager.select_eviction_candidates(
            candidates,
            required_free_mb=required_free,
        )

    @staticmethod
    def eviction_candidates_from_states(
        states: list[ExpertState],
        priorities: dict[int, float] | None = None,
    ) -> list[EvictionCandidate]:
        """
        从 ExpertState 创建 EvictionCandidate。
        """

        return CapacityManager.candidates_from_states(
            states,
            priorities=priorities,
        )
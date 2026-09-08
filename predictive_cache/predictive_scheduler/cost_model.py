from __future__ import annotations

from dataclasses import dataclass

from .models import ExpertPrediction, TransferCost


@dataclass(frozen=True)
class CostModel:
    """
    Predictive Expert Cache 的成本模型。

    核心目标：

        benefit =
            probability * reuse_cost
            - transfer_cost
            - eviction_cost

    当前阶段只负责成本计算，不负责最终 PREFETCH / WAIT / EVICT 决策。
    """

    default_reuse_cost_ms: float = 1.0

    def __post_init__(self) -> None:
        if self.default_reuse_cost_ms < 0:
            raise ValueError("default_reuse_cost_ms must be >= 0")

    def reuse_value(
        self,
        prediction: ExpertPrediction,
        reuse_cost_ms: float | None = None,
    ) -> float:
        """
        根据预测概率计算预期复用收益。

        reuse_value = probability * reuse_cost
        """
        cost = (
            self.default_reuse_cost_ms
            if reuse_cost_ms is None
            else reuse_cost_ms
        )

        if cost < 0:
            raise ValueError("reuse_cost_ms must be >= 0")

        return prediction.probability * cost

    @staticmethod
    def transfer_time_ms(
        expert_size_mb: float,
        bandwidth_mb_s: float,
    ) -> float:
        """
        根据 Expert 大小和带宽计算传输时间。

        time_ms = size_mb / bandwidth_mb_s * 1000
        """
        if expert_size_mb <= 0:
            raise ValueError("expert_size_mb must be > 0")

        if bandwidth_mb_s <= 0:
            raise ValueError("bandwidth_mb_s must be > 0")

        return expert_size_mb / bandwidth_mb_s * 1000.0

    def transfer_cost(
        self,
        prediction: ExpertPrediction,
        transfer_cost: TransferCost,
        *,
        bandwidth_mb_s: float | None = None,
    ) -> float:
        """
        计算 Expert 的传输成本。

        如果显式提供 bandwidth_mb_s，则使用：

            expert_size / bandwidth * 1000

        否则使用 TransferCost 中已经提供的传输时间。
        """
        if bandwidth_mb_s is not None:
            return self.transfer_time_ms(
                prediction.expert_size_mb,
                bandwidth_mb_s,
            )

        return transfer_cost.ram_to_gpu_ms

    @staticmethod
    def eviction_cost_ms(eviction_cost_ms: float) -> float:
        """
        标准化 eviction cost。
        """
        if eviction_cost_ms < 0:
            raise ValueError("eviction_cost_ms must be >= 0")

        return eviction_cost_ms

    def benefit(
        self,
        prediction: ExpertPrediction,
        transfer_cost: TransferCost,
        *,
        reuse_cost_ms: float | None = None,
        eviction_cost_ms: float = 0.0,
        bandwidth_mb_s: float | None = None,
    ) -> float:
        """
        计算预测缓存收益。

        benefit =
            reuse_value
            - transfer_cost
            - eviction_cost
        """
        reuse_value = self.reuse_value(
            prediction,
            reuse_cost_ms,
        )

        transfer = self.transfer_cost(
            prediction,
            transfer_cost,
            bandwidth_mb_s=bandwidth_mb_s,
        )

        eviction = self.eviction_cost_ms(
            eviction_cost_ms,
        )

        return reuse_value - transfer - eviction

    def explain(
        self,
        prediction: ExpertPrediction,
        transfer_cost: TransferCost,
        *,
        reuse_cost_ms: float | None = None,
        eviction_cost_ms: float = 0.0,
        bandwidth_mb_s: float | None = None,
    ) -> dict[str, float]:
        """
        返回完整成本计算结果。

        方便后续 Scheduler 生成可解释的决策原因。
        """
        reuse_value = self.reuse_value(
            prediction,
            reuse_cost_ms,
        )

        transfer = self.transfer_cost(
            prediction,
            transfer_cost,
            bandwidth_mb_s=bandwidth_mb_s,
        )

        eviction = self.eviction_cost_ms(
            eviction_cost_ms,
        )

        benefit = reuse_value - transfer - eviction

        return {
            "probability": prediction.probability,
            "expert_size_mb": prediction.expert_size_mb,
            "reuse_value": reuse_value,
            "transfer_cost": transfer,
            "eviction_cost": eviction,
            "benefit": benefit,
        }
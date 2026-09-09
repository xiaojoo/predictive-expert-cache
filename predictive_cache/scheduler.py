from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Mapping, Sequence

from .cache import PredictiveExpertCache
from .types import ExpertId, ExpertPrediction

from .predictive_scheduler import (
    CapacityManager,
    CostModel,
    SchedulerPolicy,
    TransferCost,
)
from .predictive_scheduler.models import ExpertPrediction as SchedulerPrediction


@dataclass(slots=True)
class PrefetchRequest:
    expert_id: ExpertId
    score: float
    priority: float


class ExpertScheduler:
    """
    High-level predictive scheduler.

    Responsibilities:
        - obtain predictions
        - remove duplicate candidates
        - skip resident experts
        - evaluate prediction benefit
        - respect GPU cache capacity
        - rank prefetch candidates
        - generate PrefetchRequest

    This class does not perform actual data movement.
    """

    def __init__(
        self,
        cache: PredictiveExpertCache,
        *,
        cost_model: CostModel | None = None,
        policy: SchedulerPolicy | None = None,
        cache_capacity_mb: float | None = None,
        cache_used_mb: float = 0.0,
        default_expert_size_mb: float = 1.0,
        default_transfer_cost: TransferCost | None = None,
        reuse_cost_ms: float = 1.0,
        minimum_benefit: float = 0.0,
    ) -> None:
        if default_expert_size_mb <= 0:
            raise ValueError(
                "default_expert_size_mb must be > 0"
            )

        if cache_capacity_mb is not None and cache_capacity_mb <= 0:
            raise ValueError(
                "cache_capacity_mb must be > 0"
            )

        if cache_used_mb < 0:
            raise ValueError(
                "cache_used_mb must be >= 0"
            )

        if reuse_cost_ms < 0:
            raise ValueError(
                "reuse_cost_ms must be >= 0"
            )

        self.cache = cache

        self.cost_model = (
            cost_model
            or CostModel(
                default_reuse_cost_ms=reuse_cost_ms,
            )
        )

        self.policy = (
            policy
            or SchedulerPolicy(
                cost_model=self.cost_model,
                minimum_benefit=minimum_benefit,
            )
        )

        self.cache_capacity_mb = cache_capacity_mb
        self.cache_used_mb = cache_used_mb
        self.default_expert_size_mb = default_expert_size_mb

        self.default_transfer_cost = (
            default_transfer_cost
            or TransferCost(
                gpu_to_ram_ms=0.0,
                ram_to_gpu_ms=0.0,
            )
        )

    # ---------------------------------------------------------
    # Legacy entry point
    # ---------------------------------------------------------

    def plan_prefetch(
        self,
        current_experts: list[ExpertId],
    ) -> list[PrefetchRequest]:
        """
        Backward-compatible prefetch planning.

        Existing callers continue to work exactly as before.
        """

        predictions = self.cache.prefetch_candidates(
            current_experts
        )

        return [
            PrefetchRequest(
                expert_id=prediction.expert_id,
                score=prediction.score,
                priority=prediction.score,
            )
            for prediction in predictions
        ]

    # ---------------------------------------------------------
    # Distance priority
    # ---------------------------------------------------------

    @staticmethod
    def _distance_factor(
        distance: float | None,
    ) -> float:
        """
        Convert estimated expert distance into a priority factor.

        Smaller distance means higher urgency.

        Examples:
            None -> 1.0
            0    -> 1.0
            1    -> 0.5
            3    -> 0.25
            9    -> 0.1

        Distance is intentionally applied only to final priority.
        It does not modify the economic benefit calculated by CostModel.
        """

        if distance is None:
            return 1.0

        if distance < 0:
            raise ValueError(
                "distance must be >= 0"
            )

        return 1.0 / (1.0 + distance)

    # ---------------------------------------------------------
    # Predictive scheduler
    # ---------------------------------------------------------

    def plan_predictions(
        self,
        predictions: Sequence[ExpertPrediction],
        *,
        expert_sizes_mb: Mapping[ExpertId, float] | None = None,
        transfer_costs: Mapping[ExpertId, TransferCost] | None = None,
        cached_experts: Iterable[ExpertId] | None = None,
        capacity_mb: float | None = None,
        used_mb: float | None = None,
        reuse_cost_ms: float | None = None,
        eviction_costs: Mapping[ExpertId, float] | None = None,
    ) -> list[PrefetchRequest]:
        """
        Convert predictions into ranked PrefetchRequests.

        The method is intentionally pure at the scheduler level:
        it does not perform cache insertion or actual data movement.

        Priority calculation:

            base_score = economic benefit

            distance_factor =
                1 / (1 + estimated_distance)

            priority =
                base_score * distance_factor

        The distance factor affects ranking priority only.
        It does not change CostModel.benefit().
        """

        expert_sizes_mb = expert_sizes_mb or {}
        transfer_costs = transfer_costs or {}
        eviction_costs = eviction_costs or {}

        if cached_experts is not None:
            resident = set(cached_experts)
        else:
            cached = getattr(self.cache, "cached", None)

            if cached is not None:
                resident = set(cached)
            else:
                resident = set(self.cache.cached_experts())

        # -----------------------------------------------------
        # Resolve capacity
        # -----------------------------------------------------

        effective_capacity = (
            self.cache_capacity_mb
            if capacity_mb is None
            else capacity_mb
        )

        effective_used = (
            self.cache_used_mb
            if used_mb is None
            else used_mb
        )

        capacity: CapacityManager | None = None

        if effective_capacity is not None:
            capacity = CapacityManager(
                capacity_mb=effective_capacity,
                used_mb=effective_used,
            )

        # -----------------------------------------------------
        # Deduplicate candidates
        # -----------------------------------------------------

        unique: dict[
            ExpertId,
            ExpertPrediction,
        ] = {}

        for prediction in predictions:
            expert_id = prediction.expert_id

            # Already resident -> no prefetch.
            if expert_id in resident:
                continue

            previous = unique.get(expert_id)

            if previous is None:
                unique[expert_id] = prediction
                continue

            # Keep the strongest prediction.
            if prediction.score > previous.score:
                unique[expert_id] = prediction

        # -----------------------------------------------------
        # Evaluate candidates
        # -----------------------------------------------------

        requests: list[PrefetchRequest] = []

        for prediction in unique.values():
            expert_id = prediction.expert_id

            expert_size_mb = expert_sizes_mb.get(
                expert_id,
                self.default_expert_size_mb,
            )

            # -------------------------------------------------
            # Build scheduler-level prediction.
            #
            # 6B-2.1c:
            # propagate estimated distance into the scheduler.
            # -------------------------------------------------

            scheduler_prediction = SchedulerPrediction(
                expert_id=prediction.expert_id,
                probability=prediction.score,
                expert_size_mb=expert_size_mb,
                distance=prediction.estimated_distance,
            )

            transfer_cost = transfer_costs.get(
                expert_id,
                self.default_transfer_cost,
            )

            eviction_cost = eviction_costs.get(
                expert_id,
                0.0,
            )

            # -------------------------------------------------
            # No capacity management.
            #
            # CostModel decides the economic benefit.
            # Distance only affects final priority.
            # -------------------------------------------------

            if capacity is None:
                benefit = self.cost_model.benefit(
                    scheduler_prediction,
                    transfer_cost,
                    reuse_cost_ms=reuse_cost_ms,
                    eviction_cost_ms=eviction_cost,
                )

                if benefit <= self.policy.minimum_benefit:
                    continue

                distance_factor = self._distance_factor(
                    scheduler_prediction.distance
                )

                priority = (
                    benefit * distance_factor
                )

                requests.append(
                    PrefetchRequest(
                        expert_id=expert_id,
                        score=benefit,
                        priority=priority,
                    )
                )

                continue

            # -------------------------------------------------
            # Capacity-aware policy.
            #
            # SchedulerPolicy returns SchedulerDecision.score
            # as the calculated benefit.
            # -------------------------------------------------

            decision = self.policy.decide(
                scheduler_prediction,
                transfer_cost,
                capacity,
                reuse_cost_ms=reuse_cost_ms,
                eviction_cost_ms=eviction_cost,
            )

            if not decision.should_prefetch:
                continue

            distance_factor = self._distance_factor(
                scheduler_prediction.distance
            )

            priority = (
                decision.score * distance_factor
            )

            requests.append(
                PrefetchRequest(
                    expert_id=expert_id,
                    score=decision.score,
                    priority=priority,
                )
            )

            # Reserve capacity immediately so that subsequent
            # candidates cannot overcommit the same cache space.
            capacity = capacity.reserve(
                expert_size_mb
            )

        # ---------------------------------------------------------
        # Highest final priority first.
        #
        # expert_id provides deterministic tie-breaking.
        # ---------------------------------------------------------

        requests.sort(
            key=lambda request: (
                -request.priority,
                request.expert_id,
            )
        )

        return requests
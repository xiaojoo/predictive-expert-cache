from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..cache import PredictiveExpertCache
from ..types import ExpertId
from .collector import RoutingEventCollector
from .qwen import (
    DefaultQwenRoutingAdapter,
    QwenRoutingAdapter,
    QwenRoutingOutput,
    collect_qwen_routing_output,
)


class QwenRoutingBridge:
    """
    Bridge Qwen routing output into the predictive cache.

    The bridge keeps model-specific routing concerns outside
    PredictiveExpertCache and PrefetchPipeline.

    Flow:

        Qwen router output
            -> QwenRoutingOutput
            -> RoutingEventCollector
            -> PredictiveExpertCache.observe()
    """

    def __init__(
        self,
        cache: PredictiveExpertCache,
        *,
        collector: RoutingEventCollector | None = None,
        adapter: QwenRoutingAdapter | None = None,
    ) -> None:
        self.cache = cache
        self.collector = (
            collector
            if collector is not None
            else RoutingEventCollector()
        )
        self.adapter = (
            adapter
            if adapter is not None
            else DefaultQwenRoutingAdapter()
        )

    def observe(
        self,
        *,
        step: int,
        output: QwenRoutingOutput,
    ) -> tuple[ExpertId, ...]:
        """
        Record one normalized Qwen routing output.

        All experts selected across the MoE layers in this output
        are deduplicated and observed as one predictor step.

        This is important because multiple layers belong to the same
        inference step and must not artificially advance predictor time.
        """

        collect_qwen_routing_output(
            self.collector,
            step=step,
            output=output,
        )

        experts: list[ExpertId] = []

        for expert_ids in output.layer_expert_ids.values():
            for expert_id in expert_ids:
                if expert_id not in experts:
                    experts.append(expert_id)

        self.cache.observe(experts)

        return tuple(experts)

    def adapt_and_observe(
        self,
        *,
        step: int,
        token_id: int,
        token_position: int,
        layer_id: int,
        router_indices,
    ) -> QwenRoutingOutput:
        """
        Adapt one real Qwen router output and feed it into the cache.
        """

        output = self.adapter.adapt_router_output(
            token_id=token_id,
            token_position=token_position,
            layer_id=layer_id,
            router_indices=router_indices,
        )

        self.observe(
            step=step,
            output=output,
        )

        return output

    def adapt_and_observe_step(
        self,
        *,
        step: int,
        token_id: int,
        token_position: int,
        layer_router_indices: Mapping[int, Any],
    ) -> QwenRoutingOutput:
        """
        Adapt all Qwen MoE layers for one inference step.

        All layers are normalized into one QwenRoutingOutput and
        observed by the predictive cache exactly once.
        """

        layer_expert_ids: dict[int, tuple[int, ...]] = {}

        for layer_id, router_indices in layer_router_indices.items():
            output = self.adapter.adapt_router_output(
                token_id=token_id,
                token_position=token_position,
                layer_id=layer_id,
                router_indices=router_indices,
            )

            layer_expert_ids[layer_id] = output.layer_expert_ids[
                layer_id
            ]

        output = QwenRoutingOutput(
            token_id=token_id,
            layer_expert_ids=layer_expert_ids,
        )

        self.observe(
            step=step,
            output=output,
        )

        return output
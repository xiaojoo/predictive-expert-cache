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

class QwenMoeRoutingCapture:
    """
    Capture real Qwen3-MoE gate outputs during model forward.

    The captureer installs forward hooks on:
        model.layers[i].mlp.gate

    Each gate returns:
        router_logits, router_scores, router_indices

    Only router_indices are retained. All MoE layers belonging to
    the same model forward are aggregated before advancing the
    predictive cache.
    """

    def __init__(
        self,
        bridge: QwenRoutingBridge,
    ) -> None:
        self.bridge = bridge
        self._hooks: list[Any] = []
        self._layer_router_indices: dict[int, Any] = {}

    @property
    def layer_router_indices(self) -> Mapping[int, Any]:
        """Return captured router indices keyed by MoE layer."""
        return dict(self._layer_router_indices)

    def install(self, model: Any) -> None:
        """Install hooks on every Qwen MoE gate in the model."""
        self.remove()

        layers = getattr(model, "layers", None)
        if layers is None:
            model_core = getattr(model, "model", None)
            layers = getattr(model_core, "layers", None)

        if layers is None:
            raise ValueError(
                "model must expose Qwen MoE layers via "
                "model.layers or model.model.layers"
            )

        for layer_id, layer in enumerate(layers):
            mlp = getattr(layer, "mlp", None)
            gate = getattr(mlp, "gate", None)

            if gate is None:
                continue

            def capture_hook(
                module: Any,
                inputs: tuple[Any, ...],
                output: Any,
                *,
                captured_layer_id: int = layer_id,
            ) -> None:
                del module, inputs

                if not isinstance(output, tuple) or len(output) != 3:
                    raise ValueError(
                        "Qwen MoE gate output must be "
                        "(router_logits, router_scores, router_indices)"
                    )

                router_indices = output[2]

                if router_indices.ndim != 2:
                    raise ValueError(
                        "Qwen router_indices must have shape "
                        "[tokens, top_k]"
                    )

                self._layer_router_indices[captured_layer_id] = (
                    router_indices.detach()
                )

            self._hooks.append(gate.register_forward_hook(capture_hook))

        if not self._hooks:
            raise ValueError("no Qwen MoE gates found in model")

    def remove(self) -> None:
        """Remove all installed hooks."""
        for hook in self._hooks:
            hook.remove()

        self._hooks.clear()

    def clear(self) -> None:
        """Clear captured routing outputs."""
        self._layer_router_indices.clear()

    def capture_forward(
        self,
        model: Any,
        *,
        input_ids: Any,
        step: int,
        **model_kwargs: Any,
    ) -> Any:
        """
        Run one real model forward and feed captured routing into cache.

        Each flattened input token becomes one predictor step.
        All MoE layers for a token are aggregated into that step.
        """
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")

        self.clear()
        self.install(model)

        try:
            output = model(
                input_ids=input_ids,
                **model_kwargs,
            )

            layer_router_indices = self.layer_router_indices

            if not layer_router_indices:
                raise RuntimeError(
                    "no Qwen MoE routing outputs were captured"
                )

            batch_size, sequence_length = input_ids.shape
            expected_tokens = batch_size * sequence_length

            for layer_id, router_indices in layer_router_indices.items():
                if router_indices.shape[0] != expected_tokens:
                    raise ValueError(
                        f"layer {layer_id} router output has "
                        f"{router_indices.shape[0]} tokens; "
                        f"expected {expected_tokens}"
                    )

            for token_position in range(expected_tokens):
                batch_index = token_position // sequence_length
                sequence_index = token_position % sequence_length
                token_id = int(
                    input_ids[batch_index, sequence_index].item()
                )

                self.bridge.adapt_and_observe_step(
                    step=step + token_position,
                    token_id=token_id,
                    token_position=token_position,
                    layer_router_indices=layer_router_indices,
                )

            return output
        finally:
            self.remove()
            self.clear()
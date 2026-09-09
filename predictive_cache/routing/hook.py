from __future__ import annotations

from typing import Protocol

from .collector import RoutingEventCollector
from .qwen import (
    QwenRoutingOutput,
    collect_qwen_routing_output,
)


class RoutingHook(Protocol):
    """Interface between a model router and the routing collector."""

    def __call__(
        self,
        *,
        step: int,
        output: QwenRoutingOutput,
        collector: RoutingEventCollector,
    ) -> None:
        ...


class QwenRoutingHook:
    """Default hook implementation for normalized Qwen routing output."""

    def __call__(
        self,
        *,
        step: int,
        output: QwenRoutingOutput,
        collector: RoutingEventCollector,
    ) -> None:
        collect_qwen_routing_output(
            collector,
            step=step,
            output=output,
        )
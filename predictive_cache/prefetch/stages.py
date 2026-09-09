from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Sequence

from .types import PrefetchTask, PrefetchTarget


StageHandler = Callable[[PrefetchTask], None]


@dataclass(frozen=True)
class PrefetchStage:
    """One physical transfer stage in a logical prefetch."""

    source: object
    target: object
    handler: StageHandler


class PrefetchStageChain:
    """Execute multiple physical transfer stages as one logical task."""

    def __init__(
        self,
        stages: Sequence[PrefetchStage],
    ) -> None:
        if not stages:
            raise ValueError("at least one prefetch stage is required")

        self._stages = tuple(stages)

    @property
    def stages(self) -> tuple[PrefetchStage, ...]:
        return self._stages

    @property
    def final_target(self) -> PrefetchTarget:
        return self._stages[-1].target

    def __call__(self, task: PrefetchTask) -> None:
        self.execute(task)

    def execute(self, task: PrefetchTask) -> None:
        current = task

        for stage in self._stages:
            current = replace(
                current,
                source=stage.source,
                target=stage.target,
            )

            stage.handler(current)
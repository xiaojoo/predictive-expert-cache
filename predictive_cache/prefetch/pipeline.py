from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..scheduler import ExpertScheduler, PrefetchRequest
from .admission import AdmissionController
from .engine import PrefetchEngine
from .types import (
    PrefetchSource,
    PrefetchTarget,
    PrefetchTask,
)


@dataclass(frozen=True)
class PipelineResult:
    """
    Result of one predictive prefetch pipeline execution.
    """

    current_experts: List[int]
    scheduled_requests: List[PrefetchRequest]
    submitted_tasks: List[PrefetchTask]


class PrefetchPipeline:
    """
    Predictive Expert Cache prefetch pipeline.

    Flow:

        ExpertScheduler
              ↓
        PrefetchRequest
              ↓
        PrefetchTask
              ↓
        AdmissionController
              ↓
        PrefetchEngine
              ↓
        Worker
              ↓
        Loader

    Admission is the gate between task construction
    and engine submission.
    """

    def __init__(
        self,
        scheduler: ExpertScheduler,
        engine: PrefetchEngine,
        *,
        source: PrefetchSource = PrefetchSource.NVME,
        target: PrefetchTarget = PrefetchTarget.RAM,
        admission: AdmissionController | None = None,
        auto_start: bool = True,
    ) -> None:
        self.scheduler = scheduler
        self.engine = engine

        self.source = source
        self.target = target

        self.admission = admission or AdmissionController()

        if auto_start:
            self.start()

    # ---------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------

    def start(self) -> None:
        self.engine.start()

    def stop(
        self,
        *,
        wait: bool = True,
    ) -> None:
        self.engine.stop(wait=wait)

    @property
    def running(self) -> bool:
        return self.engine.running

    # ---------------------------------------------------------
    # Process
    # ---------------------------------------------------------

    def process(
        self,
        current_experts: list[int],
    ) -> PipelineResult:

        requests = self.scheduler.plan_prefetch(
            current_experts
        )

        scheduled_requests = list(
            requests or []
        )

        submitted_tasks: list[PrefetchTask] = []

        for request in scheduled_requests:
            task = self._request_to_task(request)

            # -------------------------------------------------
            # Admission Gate
            # -------------------------------------------------
            if not self.admission.admit(
                task,
                self.engine,
            ):
                continue

            # -------------------------------------------------
            # Engine Submission
            # -------------------------------------------------
            if self.engine.submit(task):
                submitted_tasks.append(task)

        return PipelineResult(
            current_experts=list(current_experts),
            scheduled_requests=scheduled_requests,
            submitted_tasks=submitted_tasks,
        )

    # ---------------------------------------------------------
    # Request → Task
    # ---------------------------------------------------------

    def _request_to_task(
        self,
        request: PrefetchRequest,
    ) -> PrefetchTask:

        return PrefetchTask(
            expert_id=request.expert_id,
            source=self.source,
            target=self.target,
            priority=request.priority,
            confidence=request.score,
        )

    # ---------------------------------------------------------
    # Engine state
    # ---------------------------------------------------------

    @property
    def queue_size(self) -> int:
        return self.engine.queue_size

    def results(self):
        return self.engine.results()

    def clear_results(self) -> None:
        self.engine.clear_results()
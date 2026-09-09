from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List

from ..scheduler import ExpertScheduler, PrefetchRequest
from .admission import (
    AdmissionController,
    AdmissionDecision,
)
from .engine import PrefetchEngine
from .types import (
    PrefetchSource,
    PrefetchTarget,
    PrefetchTask,
)
from .admission_stats import AdmissionStats


@dataclass(frozen=True)
class PipelineResult:
    """
    Result of one predictive prefetch pipeline execution.
    """

    current_experts: List[int]
    scheduled_requests: List[PrefetchRequest]
    submitted_tasks: List[PrefetchTask]
    admission_decisions: List[AdmissionDecision]
    admission_stats: AdmissionStats


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
        admission_stats=None,
    ) -> None:
        self.scheduler = scheduler
        self.engine = engine

        self.source = source
        self.target = target

        self.admission = admission or AdmissionController()

        if auto_start:
            self.start()

        self.admission_stats = (
            admission_stats
            if admission_stats is not None
            else AdmissionStats()
        )

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

        planner = getattr(
            self.scheduler,
            "plan_prefetch_predictions",
            None,
        )

        if planner is not None:
            requests = planner(current_experts)
        else:
            requests = self.scheduler.plan_prefetch(
                current_experts
            )

        scheduled_requests = list(
            requests or []
        )

        submitted_tasks: list[PrefetchTask] = []
        admission_decisions: list[AdmissionDecision] = []

        for request in scheduled_requests:
            task = self._request_to_task(request)

            decision = self.admission.evaluate(
                task,
                self.engine,
            )

            admission_decisions.append(decision)
            self.admission_stats.record(decision)

            if not decision.admitted:
                continue

            if self.engine.submit(task):
                submitted_tasks.append(task)

        return PipelineResult(
            current_experts=current_experts,
            scheduled_requests=scheduled_requests,
            submitted_tasks=submitted_tasks,
            admission_decisions=admission_decisions,
            admission_stats=replace(self.admission_stats),
        )

    # ---------------------------------------------------------
    # Request → Task
    # ---------------------------------------------------------

    def _request_to_task(
            self,
            request: PrefetchRequest,
    ) -> PrefetchTask:

        confidence = getattr(
            request,
            "confidence",
            None,
        )

        estimated_distance = getattr(
            request,
            "estimated_distance",
            None,
        )

        return PrefetchTask(
            expert_id=request.expert_id,
            source=self.source,
            target=self.target,
            priority=request.priority,
            confidence=(
                confidence
                if confidence is not None
                else request.score
            ),
            estimated_distance=(
                estimated_distance
                if estimated_distance is not None
                else 0
            ),
        )

    # ---------------------------------------------------------
    # Engine state
    # ---------------------------------------------------------
    def metrics(self) -> dict[str, int | float]:
        """
        Return a detached snapshot of pipeline metrics.

        The returned dictionary is owned by the caller and does not
        expose the internal AdmissionStats instance.
        """
        return dict(self.admission_stats.metrics())


    @property
    def queue_size(self) -> int:
        return self.engine.queue_size

    def results(self):
        return self.engine.results()

    def clear_results(self) -> None:
        self.engine.clear_results()
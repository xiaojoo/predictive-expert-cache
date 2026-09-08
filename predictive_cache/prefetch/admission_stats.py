from dataclasses import dataclass

from .admission import AdmissionDecision, AdmissionReason


@dataclass
class AdmissionStats:
    total: int = 0
    accepted: int = 0
    rejected: int = 0

    reject_low_priority: int = 0
    reject_low_confidence: int = 0
    reject_duplicate_queued: int = 0
    reject_duplicate_running: int = 0
    reject_queue_full: int = 0

    def record(self, decision: AdmissionDecision) -> None:
        self.total += 1

        if decision.admitted:
            self.accepted += 1
            return

        self.rejected += 1

        counters = {
            AdmissionReason.REJECT_LOW_PRIORITY: "reject_low_priority",
            AdmissionReason.REJECT_LOW_CONFIDENCE: "reject_low_confidence",
            AdmissionReason.REJECT_DUPLICATE_QUEUED: "reject_duplicate_queued",
            AdmissionReason.REJECT_DUPLICATE_RUNNING: "reject_duplicate_running",
            AdmissionReason.REJECT_QUEUE_FULL: "reject_queue_full",
        }

        counter_name = counters.get(decision.reason)
        if counter_name is None:
            raise ValueError(
                f"Unsupported rejection reason: {decision.reason}"
            )

        setattr(
            self,
            counter_name,
            getattr(self, counter_name) + 1,
        )

    @property
    def acceptance_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.accepted / self.total

    @property
    def rejection_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.rejected / self.total

    def metrics(self) -> dict[str, int | float]:
        return {
            "admission_total": self.total,
            "admission_accepted": self.accepted,
            "admission_rejected": self.rejected,
            "admission_reject_low_priority": self.reject_low_priority,
            "admission_reject_low_confidence": self.reject_low_confidence,
            "admission_reject_duplicate_queued": self.reject_duplicate_queued,
            "admission_reject_duplicate_running": self.reject_duplicate_running,
            "admission_reject_queue_full": self.reject_queue_full,
            "admission_acceptance_rate": self.acceptance_rate,
            "admission_rejection_rate": self.rejection_rate,
        }
import pytest

from predictive_cache.prefetch.admission import (
    AdmissionDecision,
    AdmissionReason,
)
from predictive_cache.prefetch.admission_stats import AdmissionStats


def make_decision(
    admitted: bool,
    reason: AdmissionReason,
) -> AdmissionDecision:
    return AdmissionDecision(
        admitted=admitted,
        reason=reason,
    )


def test_initial_stats():
    stats = AdmissionStats()

    assert stats.total == 0
    assert stats.accepted == 0
    assert stats.rejected == 0

    assert stats.acceptance_rate == 0.0
    assert stats.rejection_rate == 0.0


def test_record_accept():
    stats = AdmissionStats()

    stats.record(
        make_decision(
            True,
            AdmissionReason.ACCEPT,
        )
    )

    assert stats.total == 1
    assert stats.accepted == 1
    assert stats.rejected == 0

    assert stats.acceptance_rate == 1.0
    assert stats.rejection_rate == 0.0


@pytest.mark.parametrize(
    ("reason", "field"),
    [
        (
            AdmissionReason.REJECT_LOW_PRIORITY,
            "reject_low_priority",
        ),
        (
            AdmissionReason.REJECT_LOW_CONFIDENCE,
            "reject_low_confidence",
        ),
        (
            AdmissionReason.REJECT_DUPLICATE_QUEUED,
            "reject_duplicate_queued",
        ),
        (
            AdmissionReason.REJECT_DUPLICATE_RUNNING,
            "reject_duplicate_running",
        ),
        (
            AdmissionReason.REJECT_QUEUE_FULL,
            "reject_queue_full",
        ),
    ],
)
def test_record_rejection_reason(reason, field):
    stats = AdmissionStats()

    stats.record(
        make_decision(
            False,
            reason,
        )
    )

    assert stats.total == 1
    assert stats.accepted == 0
    assert stats.rejected == 1
    assert getattr(stats, field) == 1

    assert stats.acceptance_rate == 0.0
    assert stats.rejection_rate == 1.0


def test_record_mixed_decisions():
    stats = AdmissionStats()

    decisions = [
        make_decision(True, AdmissionReason.ACCEPT),
        make_decision(True, AdmissionReason.ACCEPT),
        make_decision(
            False,
            AdmissionReason.REJECT_LOW_CONFIDENCE,
        ),
        make_decision(
            False,
            AdmissionReason.REJECT_LOW_PRIORITY,
        ),
        make_decision(
            False,
            AdmissionReason.REJECT_QUEUE_FULL,
        ),
    ]

    for decision in decisions:
        stats.record(decision)

    assert stats.total == 5
    assert stats.accepted == 2
    assert stats.rejected == 3

    assert stats.reject_low_confidence == 1
    assert stats.reject_low_priority == 1
    assert stats.reject_queue_full == 1

    assert stats.acceptance_rate == pytest.approx(0.4)
    assert stats.rejection_rate == pytest.approx(0.6)

    assert stats.total == stats.accepted + stats.rejected
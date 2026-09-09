from __future__ import annotations

import time
from statistics import mean, median

from predictive_cache.cache import PredictiveExpertCache


def _percentile(values: list[int], p: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    weight = rank - lo
    return ordered[lo] + (
        ordered[hi] - ordered[lo]
    ) * weight


def _report(name: str, samples: list[int]) -> None:
    values = [x / 1_000.0 for x in samples]

    print()
    print(name)
    print("-" * len(name))
    print(f"cycles:     {len(values)}")
    print(f"mean:       {mean(values):.3f} us")
    print(f"median:     {median(values):.3f} us")
    print(f"p95:        {_percentile(samples, 0.95) / 1_000.0:.3f} us")
    print(f"min:        {min(values):.3f} us")
    print(f"max:        {max(values):.3f} us")


def _make_predictor():
    cache = PredictiveExpertCache()

    for expert_id in range(64):
        cache.observe([expert_id])

    return cache.predictor


def test_7f_3_predictor_internal_attribution() -> None:
    cycles = 10_000
    predictor = _make_predictor()

    stats = predictor._stats
    assert stats

    candidate = next(iter(stats))

    frequency_samples: list[int] = []
    recency_samples: list[int] = []
    transition_samples: list[int] = []
    combined_samples: list[int] = []

    previous = candidate

    for _ in range(cycles):
        started = time.perf_counter_ns()

        value = predictor._frequency_score(candidate)

        frequency_samples.append(
            time.perf_counter_ns() - started
        )

        assert value >= 0.0

    for _ in range(cycles):
        started = time.perf_counter_ns()

        value = predictor._recency_score(candidate)

        recency_samples.append(
            time.perf_counter_ns() - started
        )

        assert value >= 0.0

    for _ in range(cycles):
        started = time.perf_counter_ns()

        value = predictor._transition_score(
            previous,
            candidate,
        )

        transition_samples.append(
            time.perf_counter_ns() - started
        )

        assert value >= 0.0

    for _ in range(cycles):
        started = time.perf_counter_ns()

        frequency = predictor._frequency_score(candidate)
        recency = predictor._recency_score(candidate)
        transition = predictor._transition_score(
            previous,
            candidate,
        )

        score = (
            predictor.frequency_weight * frequency
            + predictor.recency_weight * recency
            + predictor.transition_weight * transition
        )

        combined_samples.append(
            time.perf_counter_ns() - started
        )

        assert score >= 0.0

    _report(
        "7-F-3 Predictor._frequency_score()",
        frequency_samples,
    )

    _report(
        "7-F-3 Predictor._recency_score()",
        recency_samples,
    )

    _report(
        "7-F-3 Predictor._transition_score()",
        transition_samples,
    )

    _report(
        "7-F-3 Predictor combined scoring",
        combined_samples,
    )


if __name__ == "__main__":
    test_7f_3_predictor_internal_attribution()

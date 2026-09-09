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
    ) * (rank - lo)


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


def _benchmark(predictor, cycles: int = 10_000) -> list[int]:
    samples: list[int] = []

    for i in range(cycles):
        current = [i % 64]

        started = time.perf_counter_ns()

        result = predictor.predict(current)

        samples.append(
            time.perf_counter_ns() - started
        )

        assert result

    return samples


def test_7f_4_predictor_score_attribution() -> None:
    cycles = 10_000

    predictor = _make_predictor()

    original_frequency = predictor._frequency_score
    original_recency = predictor._recency_score
    original_transition = predictor._transition_score

    try:
        # Baseline: all scoring enabled.
        baseline = _benchmark(predictor, cycles)

        # Disable frequency scoring.
        predictor._frequency_score = lambda candidate: 0.0
        without_frequency = _benchmark(
            predictor,
            cycles,
        )

        # Restore frequency; disable recency.
        predictor._frequency_score = original_frequency
        predictor._recency_score = lambda candidate: 0.0
        without_recency = _benchmark(
            predictor,
            cycles,
        )

        # Restore recency; disable transition.
        predictor._recency_score = original_recency
        predictor._transition_score = (
            lambda previous, candidate: 0.0
        )
        without_transition = _benchmark(
            predictor,
            cycles,
        )

        _report(
            "7-F-4 Predictor.predict() baseline",
            baseline,
        )

        _report(
            "7-F-4 Predictor.predict() without frequency",
            without_frequency,
        )

        _report(
            "7-F-4 Predictor.predict() without recency",
            without_recency,
        )

        _report(
            "7-F-4 Predictor.predict() without transition",
            without_transition,
        )

    finally:
        predictor._frequency_score = original_frequency
        predictor._recency_score = original_recency
        predictor._transition_score = original_transition


if __name__ == "__main__":
    test_7f_4_predictor_score_attribution()

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


def _make_predictor(size: int):
    cache = PredictiveExpertCache()

    for expert_id in range(size):
        cache.observe([expert_id])

    return cache.predictor


def _measure_predict(predictor, expert_id: int, cycles: int):
    samples: list[int] = []

    for _ in range(cycles):
        started = time.perf_counter_ns()

        result = predictor.predict([expert_id])

        samples.append(
            time.perf_counter_ns() - started
        )

        assert result

    return samples


def test_7f_5_predictor_scaling() -> None:
    cycles = 2_000

    sizes = [16, 32, 64, 128, 256]

    medians: list[tuple[int, float]] = []

    for size in sizes:
        predictor = _make_predictor(size)

        samples = _measure_predict(
            predictor,
            0,
            cycles,
        )

        _report(
            f"7-F-5 Predictor.predict() candidates={size}",
            samples,
        )

        medians.append(
            (size, median([x / 1_000.0 for x in samples]))
        )

    print()
    print("7-F-5 Scaling summary")
    print("---------------------")

    for size, value in medians:
        print(
            f"candidates={size:3d}  "
            f"median={value:8.3f} us"
        )


if __name__ == "__main__":
    test_7f_5_predictor_scaling()

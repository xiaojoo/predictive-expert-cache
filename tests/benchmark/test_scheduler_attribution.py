from __future__ import annotations

import time
from statistics import mean, median

from predictive_cache.cache import PredictiveExpertCache
from predictive_cache.scheduler import ExpertScheduler


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
    print(f"p95:        {_percentile(values, 0.95):.3f} us")
    print(f"min:        {min(values):.3f} us")
    print(f"max:        {max(values):.3f} us")


def _make_scheduler() -> ExpertScheduler:
    cache = PredictiveExpertCache()

    for expert_id in range(64):
        cache.observe([expert_id])

    return ExpertScheduler(
        cache,
        minimum_benefit=0.0,
    )


def test_7f_scheduler_attribution() -> None:
    cycles = 10_000

    scheduler = _make_scheduler()
    cache = scheduler.cache
    predictor = cache.predictor

    predictor_samples: list[int] = []
    candidates_samples: list[int] = []
    plan_samples: list[int] = []
    full_samples: list[int] = []

    # Freeze representative inputs so each component measures
    # the same logical workload.
    current_experts = [0]

    predictions = cache.predict(
        current_experts
    )
    assert predictions

    try:
        # -----------------------------------------------------
        # 1. Predictor only
        # -----------------------------------------------------
        for _ in range(cycles):
            started = time.perf_counter_ns()

            result = predictor.predict(
                current_experts,
                top_k=cache.config.prediction_top_k,
            )

            predictor_samples.append(
                time.perf_counter_ns() - started
            )

            assert result

        # -----------------------------------------------------
        # 2. Cache candidate generation
        #    predict + resident filtering
        # -----------------------------------------------------
        for _ in range(cycles):
            started = time.perf_counter_ns()

            result = cache.prefetch_candidates(
                current_experts
            )

            candidates_samples.append(
                time.perf_counter_ns() - started
            )

            assert result

        # -----------------------------------------------------
        # 3. Scheduler policy only
        #    Feed precomputed predictions.
        # -----------------------------------------------------
        for _ in range(cycles):
            started = time.perf_counter_ns()

            result = scheduler.plan_predictions(
                predictions
            )

            plan_samples.append(
                time.perf_counter_ns() - started
            )

            assert result

        # -----------------------------------------------------
        # 4. Full scheduler path
        # -----------------------------------------------------
        for _ in range(cycles):
            started = time.perf_counter_ns()

            result = scheduler.plan_prefetch_predictions(
                current_experts
            )

            full_samples.append(
                time.perf_counter_ns() - started
            )

            assert result

        _report(
            "7-F-2 ExpertPredictor.predict()",
            predictor_samples,
        )

        _report(
            "7-F-2 PredictiveExpertCache.prefetch_candidates()",
            candidates_samples,
        )

        _report(
            "7-F-2 ExpertScheduler.plan_predictions()",
            plan_samples,
        )

        _report(
            "7-F-2 ExpertScheduler.plan_prefetch_predictions()",
            full_samples,
        )

    finally:
        pass


if __name__ == "__main__":
    test_7f_scheduler_attribution()

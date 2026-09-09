from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, median

from predictive_cache.predictor import ExpertPredictor


def _benchmark(fn, cycles: int = 10_000) -> tuple[float, float, float, float, float]:
    samples = []

    for _ in range(cycles):
        start = __import__("time").perf_counter_ns()
        fn()
        end = __import__("time").perf_counter_ns()
        samples.append((end - start) / 1_000.0)

    samples.sort()

    p95_index = int(len(samples) * 0.95)
    if p95_index >= len(samples):
        p95_index = len(samples) - 1

    return (
        mean(samples),
        median(samples),
        samples[p95_index],
        min(samples),
        max(samples),
    )


def _print_result(name: str, result: tuple[float, float, float, float, float]) -> None:
    avg, med, p95, minimum, maximum = result

    print(name)
    print("-" * len(name))
    print("cycles:", "    10000")
    print(f"mean:       {avg:.3f} us")
    print(f"median:     {med:.3f} us")
    print(f"p95:        {p95:.3f} us")
    print(f"min:        {minimum:.3f} us")
    print(f"max:        {maximum:.3f} us")
    print()


def _build_predictor() -> ExpertPredictor:
    predictor = ExpertPredictor(
        recent_window=32,
        frequency_weight=0.30,
        recency_weight=0.20,
        transition_weight=0.50,
        recency_decay=8.0,
    )

    # Build enough transition history to make the transition
    # counters representative of the production predictor path.
    for step in range(128):
        current = [
            step % 64,
            (step + 1) % 64,
            (step + 2) % 64,
            (step + 3) % 64,
        ]
        predictor.observe(current)

    return predictor


def test_predictor_transition_attribution() -> None:
    predictor = _build_predictor()

    current_experts = [1, 2, 3, 4]
    candidate = 32

    transitions = predictor._transitions[1]

    def transition_probability() -> float:
        return predictor.transition_probability(
            1,
            candidate,
        )

    def transition_counter_get() -> int:
        return transitions.get(
            candidate,
            0,
        )

    def transition_total() -> int:
        return sum(
            transitions.values()
        )

    def transition_probability_manual() -> float:
        count = transitions.get(
            candidate,
            0,
        )
        total = sum(
            transitions.values()
        )

        if total == 0:
            return 0.0

        return count / total

    def transition_score() -> float:
        return predictor._transition_score(
            current_experts,
            candidate,
        )

    result = _benchmark(
        transition_probability,
    )
    _print_result(
        "7-F-7 Predictor.transition_probability()",
        result,
    )

    result = _benchmark(
        transition_counter_get,
    )
    _print_result(
        "7-F-7 Transition Counter.get()",
        result,
    )

    result = _benchmark(
        transition_total,
    )
    _print_result(
        "7-F-7 sum(transitions.values())",
        result,
    )

    result = _benchmark(
        transition_probability_manual,
    )
    _print_result(
        "7-F-7 Manual transition probability",
        result,
    )

    result = _benchmark(
        transition_score,
    )
    _print_result(
        "7-F-7 Predictor._transition_score()",
        result,
    )


def test_predictor_transition_total_scaling() -> None:
    predictor = _build_predictor()

    transition_counts = defaultdict(Counter)

    for source in range(4):
        counter = transition_counts[source]

        for candidate in range(256):
            counter[candidate] = candidate + 1

    predictor._transitions = transition_counts

    print("7-F-7 Transition total scaling")
    print("--------------------------------")

    for width in (16, 32, 64, 128, 256):
        transitions = transition_counts[0]

        def total() -> int:
            return sum(
                transitions[candidate]
                for candidate in range(width)
            )

        result = _benchmark(total)

        avg, med, p95, minimum, maximum = result

        print(
            f"width={width:3d} "
            f"median={med:7.3f} us "
            f"mean={avg:7.3f} us "
            f"p95={p95:7.3f} us"
        )

    print()
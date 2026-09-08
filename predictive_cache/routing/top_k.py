from __future__ import annotations

from .scoring import PredictionScore


def select_top_k_predictions(
    predictions: dict[int, PredictionScore],
    top_k: int,
    min_score: float,
) -> tuple[PredictionScore, ...]:
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    if not 0.0 <= min_score <= 1.0:
        raise ValueError("min_score must be between 0 and 1")

    candidates = (
        prediction
        for prediction in predictions.values()
        if prediction.score >= min_score
    )

    ordered = sorted(
        candidates,
        key=lambda prediction: (
            -prediction.score,
            prediction.expert_id,
        ),
    )

    return tuple(ordered[:top_k])
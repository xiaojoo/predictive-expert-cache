from __future__ import annotations

from collections.abc import Mapping, Sequence

from .models import ExpertPrediction
from predictive_cache.routing.scoring import PredictionScore


def prediction_score_to_scheduler_prediction(
    prediction: PredictionScore,
    expert_size_mb: float,
) -> ExpertPrediction:
    """
    Convert a routing PredictionScore into a scheduler ExpertPrediction.

    The routing layer owns prediction scoring.

    The scheduler layer expects:

        expert_id
        probability
        expert_size_mb

    The prediction score is used directly as the scheduler probability.
    Expert size must be supplied explicitly by the caller.
    """
    if expert_size_mb <= 0:
        raise ValueError("expert_size_mb must be > 0")

    return ExpertPrediction(
        expert_id=prediction.expert_id,
        probability=prediction.score,
        expert_size_mb=expert_size_mb,
    )


def prediction_scores_to_scheduler_predictions(
    predictions: Sequence[PredictionScore],
    expert_sizes_mb: Mapping[int, float],
) -> tuple[ExpertPrediction, ...]:
    """
    Convert ordered routing predictions into scheduler predictions.

    The input order is preserved.
    """
    result: list[ExpertPrediction] = []

    for prediction in predictions:
        try:
            expert_size_mb = expert_sizes_mb[prediction.expert_id]
        except KeyError as exc:
            raise KeyError(
                f"missing expert size for expert_id={prediction.expert_id}"
            ) from exc

        result.append(
            prediction_score_to_scheduler_prediction(
                prediction,
                expert_size_mb=expert_size_mb,
            )
        )

    return tuple(result)
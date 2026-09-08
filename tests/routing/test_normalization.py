from predictive_cache.routing.features import (
    ExpertCandidateFeatures,
)
from predictive_cache.routing.normalization import (
    NormalizedExpertFeatures,
    normalize_features,
)


def make_features(
    expert_id: int,
    frequency: int,
    recency: int,
    transition_probability: float,
) -> ExpertCandidateFeatures:
    return ExpertCandidateFeatures(
        expert_id=expert_id,
        frequency=frequency,
        first_seen_step=0,
        last_seen_step=10,
        recency=recency,
        layers=frozenset({0}),
        transition_probability=transition_probability,
    )


def test_frequency_is_normalized_by_max_frequency():
    features = {
        1: make_features(
            expert_id=1,
            frequency=10,
            recency=0,
            transition_probability=0.5,
        ),
        2: make_features(
            expert_id=2,
            frequency=5,
            recency=2,
            transition_probability=0.3,
        ),
        3: make_features(
            expert_id=3,
            frequency=2,
            recency=4,
            transition_probability=0.1,
        ),
    }

    normalized = normalize_features(features)

    assert normalized[1].frequency_score == 1.0
    assert normalized[2].frequency_score == 0.5
    assert normalized[3].frequency_score == 0.2


def test_single_expert_frequency_score_is_one():
    features = {
        7: make_features(
            expert_id=7,
            frequency=42,
            recency=3,
            transition_probability=0.8,
        ),
    }

    normalized = normalize_features(features)

    assert normalized[7].frequency_score == 1.0


def test_transition_probability_is_preserved():
    features = {
        1: make_features(
            expert_id=1,
            frequency=10,
            recency=0,
            transition_probability=0.75,
        ),
        2: make_features(
            expert_id=2,
            frequency=5,
            recency=1,
            transition_probability=0.25,
        ),
    }

    normalized = normalize_features(features)

    assert normalized[1].transition_probability == 0.75
    assert normalized[2].transition_probability == 0.25


def test_recency_is_preserved():
    features = {
        1: make_features(
            expert_id=1,
            frequency=10,
            recency=0,
            transition_probability=0.5,
        ),
        2: make_features(
            expert_id=2,
            frequency=5,
            recency=7,
            transition_probability=0.3,
        ),
    }

    normalized = normalize_features(features)

    assert normalized[1].recency == 0
    assert normalized[2].recency == 7


def test_empty_features_return_empty_result():
    normalized = normalize_features({})

    assert normalized == {}


def test_expert_ids_are_preserved():
    features = {
        3: make_features(
            expert_id=3,
            frequency=8,
            recency=2,
            transition_probability=0.4,
        ),
        9: make_features(
            expert_id=9,
            frequency=4,
            recency=5,
            transition_probability=0.2,
        ),
    }

    normalized = normalize_features(features)

    assert set(normalized) == {3, 9}
    assert normalized[3].expert_id == 3
    assert normalized[9].expert_id == 9


def test_frequency_scores_are_within_zero_and_one():
    features = {
        1: make_features(
            expert_id=1,
            frequency=100,
            recency=0,
            transition_probability=1.0,
        ),
        2: make_features(
            expert_id=2,
            frequency=1,
            recency=100,
            transition_probability=0.0,
        ),
    }

    normalized = normalize_features(features)

    for feature in normalized.values():
        assert 0.0 <= feature.frequency_score <= 1.0


def test_normalized_features_have_expected_type():
    features = {
        1: make_features(
            expert_id=1,
            frequency=10,
            recency=2,
            transition_probability=0.6,
        ),
    }

    normalized = normalize_features(features)

    assert isinstance(
        normalized[1],
        NormalizedExpertFeatures,
    )
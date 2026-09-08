import pytest

from predictive_cache.routing import (
    ExpertCandidateFeatures,
    ExpertPredictionFeatures,
    RoutingEvent,
    RoutingHistory,
    build_candidate_features,
    build_expert_features,
    build_expert_transitions,
    build_transition_probabilities,
    get_transition_probability,
)


def make_history() -> RoutingHistory:
    return RoutingHistory(
        (
            RoutingEvent(
                step=0,
                token_id=0,
                layer_id=0,
                expert_ids=(3, 7),
            ),
            RoutingEvent(
                step=1,
                token_id=1,
                layer_id=0,
                expert_ids=(3, 9),
            ),
            RoutingEvent(
                step=2,
                token_id=2,
                layer_id=1,
                expert_ids=(3, 7),
            ),
            RoutingEvent(
                step=5,
                token_id=3,
                layer_id=1,
                expert_ids=(7, 11),
            ),
        )
    )


def test_prediction_features():
    history = make_history()

    features = build_expert_features(
        history,
        current_step=10,
    )

    expert = features[3]

    assert expert.expert_id == 3
    assert expert.frequency == 3
    assert expert.first_seen_step == 0
    assert expert.last_seen_step == 2
    assert expert.recency == 8
    assert expert.layers == frozenset({0, 1})


def test_prediction_features_for_each_expert():
    history = make_history()

    features = build_expert_features(
        history,
        current_step=10,
    )

    assert set(features) == {3, 7, 9, 11}


def test_recency_is_current_step_minus_last_seen():
    history = make_history()

    features = build_expert_features(
        history,
        current_step=8,
    )

    assert features[7].last_seen_step == 5
    assert features[7].recency == 3


def test_recency_zero_for_currently_seen_expert():
    history = make_history()

    features = build_expert_features(
        history,
        current_step=5,
    )

    assert features[7].recency == 0


def test_current_step_cannot_be_before_last_seen():
    history = make_history()

    with pytest.raises(ValueError, match="current_step"):
        build_expert_features(
            history,
            current_step=4,
        )


def test_negative_current_step_rejected():
    history = make_history()

    with pytest.raises(ValueError, match="current_step"):
        build_expert_features(
            history,
            current_step=-1,
        )


def test_get_transition_probability():
    history = make_history()

    transitions = build_expert_transitions(history)

    probabilities = build_transition_probabilities(
        transitions
    )

    probability = get_transition_probability(
        probabilities,
        current_experts=(3,),
        candidate_expert_id=7,
    )

    assert probability == pytest.approx(2 / 3)


def test_get_transition_probability_uses_strongest_source():
    history = RoutingHistory(
        (
            RoutingEvent(
                step=0,
                token_id=0,
                layer_id=0,
                expert_ids=(3, 7),
            ),
            RoutingEvent(
                step=1,
                token_id=1,
                layer_id=0,
                expert_ids=(3, 9),
            ),
            RoutingEvent(
                step=2,
                token_id=2,
                layer_id=0,
                expert_ids=(5, 7),
            ),
            RoutingEvent(
                step=3,
                token_id=3,
                layer_id=0,
                expert_ids=(5, 7),
            ),
        )
    )

    transitions = build_expert_transitions(history)

    probabilities = build_transition_probabilities(
        transitions
    )

    probability = get_transition_probability(
        probabilities,
        current_experts=(3, 5),
        candidate_expert_id=7,
    )

    assert probability == pytest.approx(1.0)


def test_missing_transition_returns_zero():
    history = make_history()

    transitions = build_expert_transitions(history)

    probabilities = build_transition_probabilities(
        transitions
    )

    probability = get_transition_probability(
        probabilities,
        current_experts=(3,),
        candidate_expert_id=11,
    )

    assert probability == 0.0


def test_negative_candidate_expert_rejected():
    with pytest.raises(
        ValueError,
        match="candidate_expert_id",
    ):
        get_transition_probability(
            {},
            current_experts=(3,),
            candidate_expert_id=-1,
        )


def test_negative_current_expert_rejected():
    with pytest.raises(
        ValueError,
        match="current expert id",
    ):
        get_transition_probability(
            {},
            current_experts=(-1,),
            candidate_expert_id=7,
        )


def test_candidate_features_combine_history_and_transition():
    history = make_history()

    transitions = build_expert_transitions(history)

    probabilities = build_transition_probabilities(
        transitions
    )

    features = build_candidate_features(
        history,
        current_step=10,
        current_experts=(3,),
        transition_probabilities=probabilities,
    )

    expert = features[7]

    assert expert.expert_id == 7
    assert expert.frequency == 3
    assert expert.first_seen_step == 0
    assert expert.last_seen_step == 5
    assert expert.recency == 5
    assert expert.layers == frozenset({0, 1})
    assert expert.transition_probability == pytest.approx(2 / 3)


def test_candidate_without_transition_has_zero_probability():
    history = make_history()

    transitions = build_expert_transitions(history)

    probabilities = build_transition_probabilities(
        transitions
    )

    features = build_candidate_features(
        history,
        current_step=10,
        current_experts=(3,),
        transition_probabilities=probabilities,
    )

    assert features[11].transition_probability == 0.0


def test_candidate_features_contains_all_known_experts():
    history = make_history()

    transitions = build_expert_transitions(history)

    probabilities = build_transition_probabilities(
        transitions
    )

    features = build_candidate_features(
        history,
        current_step=10,
        current_experts=(3,),
        transition_probabilities=probabilities,
    )

    assert set(features) == {3, 7, 9, 11}


def test_prediction_features_validation():
    with pytest.raises(
        ValueError,
        match="frequency",
    ):
        ExpertPredictionFeatures(
            expert_id=1,
            frequency=0,
            first_seen_step=0,
            last_seen_step=0,
            recency=0,
            layers=frozenset({0}),
        )


def test_candidate_features_validation():
    with pytest.raises(
        ValueError,
        match="transition_probability",
    ):
        ExpertCandidateFeatures(
            expert_id=1,
            frequency=1,
            first_seen_step=0,
            last_seen_step=0,
            recency=0,
            layers=frozenset({0}),
            transition_probability=1.1,
        )


def test_candidate_features_negative_recency_rejected():
    with pytest.raises(
        ValueError,
        match="recency",
    ):
        ExpertCandidateFeatures(
            expert_id=1,
            frequency=1,
            first_seen_step=0,
            last_seen_step=0,
            recency=-1,
            layers=frozenset({0}),
            transition_probability=0.0,
        )
from predictive_cache.predictor import ExpertPredictor


def test_frequency():
    predictor = ExpertPredictor()

    for _ in range(10):
        predictor.observe([1])

    for _ in range(5):
        predictor.observe([2])

    assert predictor.stats[1].frequency == 10
    assert predictor.stats[2].frequency == 5


def test_transition_probability():
    predictor = ExpertPredictor()

    predictor.observe([1])
    predictor.observe([2])
    predictor.observe([1])
    predictor.observe([2])

    probability = predictor.transition_probability(1, 2)

    assert probability == 1.0


def test_prediction():
    predictor = ExpertPredictor(
        frequency_weight=0.2,
        recency_weight=0.1,
        transition_weight=0.7,
    )

    for _ in range(10):
        predictor.observe([1])

    for _ in range(3):
        predictor.observe([2])

    predictor.observe([1])
    predictor.observe([3])

    predictions = predictor.predict(
        current_experts=[1],
        top_k=3,
    )

    assert predictions

    ids = [prediction.expert_id for prediction in predictions]

    assert 3 in ids


def test_prediction_ranking():
    """
    Synthetic MoE routing pattern:

        1 -> 2
        1 -> 2
        1 -> 2
        1 -> 3
        1 -> 2

    Expert 2 should have a stronger transition probability
    from Expert 1 than Expert 3.
    """

    predictor = ExpertPredictor(
        frequency_weight=0.0,
        recency_weight=0.0,
        transition_weight=1.0,
    )

    predictor.observe([1])
    predictor.observe([2])

    predictor.observe([1])
    predictor.observe([2])

    predictor.observe([1])
    predictor.observe([2])

    predictor.observe([1])
    predictor.observe([3])

    predictor.observe([1])
    predictor.observe([2])

    predictions = predictor.predict(
        current_experts=[1],
        top_k=2,
    )

    assert len(predictions) == 2

    assert predictions[0].expert_id == 2
    assert predictions[1].expert_id == 3

    assert predictions[0].transition_score > predictions[1].transition_score


def test_transition_distribution():
    """
    Verify:

        1 -> 2 : 4 times
        1 -> 3 : 1 time

    Therefore:

        P(2 | 1) = 0.8
        P(3 | 1) = 0.2
    """

    predictor = ExpertPredictor()

    predictor.observe([1])
    predictor.observe([2])

    predictor.observe([1])
    predictor.observe([2])

    predictor.observe([1])
    predictor.observe([2])

    predictor.observe([1])
    predictor.observe([2])

    predictor.observe([1])
    predictor.observe([3])

    probability_2 = predictor.transition_probability(1, 2)
    probability_3 = predictor.transition_probability(1, 3)

    assert probability_2 == 0.8
    assert probability_3 == 0.2


def test_recent_window():
    predictor = ExpertPredictor(recent_window=3)

    predictor.observe([1])
    predictor.observe([2])
    predictor.observe([3])
    predictor.observe([4])

    assert predictor.recent == [2, 3, 4]

def test_prediction_tie_break_is_deterministic():
    predictor = ExpertPredictor(
        frequency_weight=1.0,
        recency_weight=0.0,
        transition_weight=0.0,
    )

    predictor.observe([1])
    predictor.observe([2, 3])

    predictions = predictor.predict(
        [1],
        top_k=2,
    )

    assert [
        prediction.expert_id
        for prediction in predictions
    ] == [2, 3]


def test_duplicate_experts_in_one_router_event_are_counted_once():
    predictor = ExpertPredictor()

    predictor.observe(
        [1, 1, 2, 2]
    )

    assert predictor.stats[1].frequency == 1
    assert predictor.stats[2].frequency == 1
    assert predictor.recent == [1, 2]

def test_expert_distance():
    predictor = ExpertPredictor()

    predictor.observe([1])
    predictor.observe([2])
    predictor.observe([3])
    predictor.observe([1])

    stats = predictor.stats[1]

    assert stats.distance_count == 1
    assert stats.distance_sum == 3
    assert stats.average_distance == 3.0

def test_prediction_contains_historical_distance():
    predictor = ExpertPredictor()

    predictor.observe([1])
    predictor.observe([2])
    predictor.observe([1])

    predictions = predictor.predict(
        current_experts=[2],
        top_k=1,
    )

    assert len(predictions) == 1
    assert predictions[0].expert_id == 1
    assert predictions[0].distance == 2.0
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

    probability = predictor.transition_probability(
        1,
        2,
    )

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

    ids = [
        prediction.expert_id
        for prediction in predictions
    ]

    assert 3 in ids


def test_recent_window():

    predictor = ExpertPredictor(
        recent_window=3
    )

    predictor.observe([1])
    predictor.observe([2])
    predictor.observe([3])
    predictor.observe([4])

    assert predictor.recent == [2, 3, 4]
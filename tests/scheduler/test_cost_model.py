import pytest

from predictive_cache.predictive_scheduler import (
    CostModel,
    ExpertPrediction,
    TransferCost,
)


def test_reuse_value():
    model = CostModel(default_reuse_cost_ms=100.0)

    prediction = ExpertPrediction(
        expert_id=17,
        probability=0.8,
        expert_size_mb=512.0,
    )

    value = model.reuse_value(prediction)

    assert value == pytest.approx(80.0)


def test_reuse_value_custom_cost():
    model = CostModel(default_reuse_cost_ms=100.0)

    prediction = ExpertPrediction(
        expert_id=17,
        probability=0.75,
        expert_size_mb=512.0,
    )

    value = model.reuse_value(
        prediction,
        reuse_cost_ms=200.0,
    )

    assert value == pytest.approx(150.0)


def test_transfer_time_from_bandwidth():
    model = CostModel()

    # 512 MB / 1024 MB/s = 0.5 s = 500 ms
    result = model.transfer_time_ms(
        expert_size_mb=512.0,
        bandwidth_mb_s=1024.0,
    )

    assert result == pytest.approx(500.0)


def test_transfer_cost_uses_explicit_bandwidth():
    model = CostModel()

    prediction = ExpertPrediction(
        expert_id=17,
        probability=0.8,
        expert_size_mb=512.0,
    )

    transfer = TransferCost(
        gpu_to_ram_ms=100.0,
        ram_to_gpu_ms=200.0,
    )

    result = model.transfer_cost(
        prediction,
        transfer,
        bandwidth_mb_s=1024.0,
    )

    assert result == pytest.approx(500.0)


def test_transfer_cost_uses_transfer_model_without_bandwidth():
    model = CostModel()

    prediction = ExpertPrediction(
        expert_id=17,
        probability=0.8,
        expert_size_mb=512.0,
    )

    transfer = TransferCost(
        gpu_to_ram_ms=100.0,
        ram_to_gpu_ms=250.0,
    )

    result = model.transfer_cost(
        prediction,
        transfer,
    )

    assert result == pytest.approx(250.0)


def test_benefit():
    model = CostModel()

    prediction = ExpertPrediction(
        expert_id=17,
        probability=0.9,
        expert_size_mb=512.0,
    )

    transfer = TransferCost(
        gpu_to_ram_ms=20.0,
        ram_to_gpu_ms=30.0,
    )

    result = model.benefit(
        prediction,
        transfer,
        reuse_cost_ms=100.0,
        eviction_cost_ms=10.0,
    )

    # 0.9 * 100 - 30 - 10 = 50
    assert result == pytest.approx(50.0)


def test_benefit_with_bandwidth():
    model = CostModel()

    prediction = ExpertPrediction(
        expert_id=17,
        probability=0.9,
        expert_size_mb=512.0,
    )

    transfer = TransferCost(
        gpu_to_ram_ms=20.0,
        ram_to_gpu_ms=30.0,
    )

    result = model.benefit(
        prediction,
        transfer,
        reuse_cost_ms=1000.0,
        eviction_cost_ms=50.0,
        bandwidth_mb_s=1024.0,
    )

    # reuse = 0.9 * 1000 = 900
    # transfer = 512 / 1024 * 1000 = 500
    # eviction = 50
    # benefit = 900 - 500 - 50 = 350
    assert result == pytest.approx(350.0)


def test_explain():
    model = CostModel()

    prediction = ExpertPrediction(
        expert_id=17,
        probability=0.8,
        expert_size_mb=512.0,
    )

    transfer = TransferCost(
        gpu_to_ram_ms=20.0,
        ram_to_gpu_ms=100.0,
    )

    result = model.explain(
        prediction,
        transfer,
        reuse_cost_ms=500.0,
        eviction_cost_ms=25.0,
    )

    assert result["probability"] == pytest.approx(0.8)
    assert result["expert_size_mb"] == pytest.approx(512.0)
    assert result["reuse_value"] == pytest.approx(400.0)
    assert result["transfer_cost"] == pytest.approx(100.0)
    assert result["eviction_cost"] == pytest.approx(25.0)
    assert result["benefit"] == pytest.approx(275.0)


def test_negative_reuse_cost_rejected():
    model = CostModel()

    prediction = ExpertPrediction(
        expert_id=1,
        probability=0.5,
        expert_size_mb=100.0,
    )

    with pytest.raises(ValueError):
        model.reuse_value(
            prediction,
            reuse_cost_ms=-1.0,
        )


def test_invalid_bandwidth_rejected():
    model = CostModel()

    with pytest.raises(ValueError):
        model.transfer_time_ms(
            expert_size_mb=100.0,
            bandwidth_mb_s=0.0,
        )


def test_negative_eviction_cost_rejected():
    with pytest.raises(ValueError):
        CostModel.eviction_cost_ms(-1.0)
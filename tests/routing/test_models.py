import pytest

from predictive_cache.routing import RoutingEvent


def test_routing_event_creation() -> None:
    event = RoutingEvent(
        step=100,
        token_id=1000,
        layer_id=12,
        expert_ids=(3, 17),
    )

    assert event.step == 100
    assert event.token_id == 1000
    assert event.layer_id == 12
    assert event.expert_ids == (3, 17)


def test_routing_event_is_immutable() -> None:
    event = RoutingEvent(
        step=1,
        token_id=2,
        layer_id=3,
        expert_ids=(4, 5),
    )

    with pytest.raises(AttributeError):
        event.step = 10  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "step": -1,
            "token_id": 0,
            "layer_id": 0,
            "expert_ids": (1,),
        },
        {
            "step": 0,
            "token_id": -1,
            "layer_id": 0,
            "expert_ids": (1,),
        },
        {
            "step": 0,
            "token_id": 0,
            "layer_id": -1,
            "expert_ids": (1,),
        },
    ],
)
def test_routing_event_rejects_negative_fields(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        RoutingEvent(**kwargs)


def test_routing_event_requires_experts() -> None:
    with pytest.raises(ValueError, match="expert_ids must not be empty"):
        RoutingEvent(
            step=0,
            token_id=0,
            layer_id=0,
            expert_ids=(),
        )


def test_routing_event_rejects_negative_expert_id() -> None:
    with pytest.raises(ValueError, match="expert_id must be >= 0"):
        RoutingEvent(
            step=0,
            token_id=0,
            layer_id=0,
            expert_ids=(1, -2),
        )


def test_routing_event_rejects_duplicate_experts() -> None:
    with pytest.raises(
        ValueError,
        match="expert_ids must not contain duplicates",
    ):
        RoutingEvent(
            step=0,
            token_id=0,
            layer_id=0,
            expert_ids=(3, 17, 17),
        )


def test_routing_event_supports_top_k_routing() -> None:
    event = RoutingEvent(
        step=42,
        token_id=1234,
        layer_id=8,
        expert_ids=(3, 7, 19, 41),
    )

    assert len(event.expert_ids) == 4

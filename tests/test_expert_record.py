from predictive_cache.storage import ExpertLocation, ExpertRecord


def test_expert_record_stores_identity_location_and_payload():
    payload = {"weights": "dummy"}

    record = ExpertRecord(
        expert_id="expert-a",
        location=ExpertLocation.RAM,
        payload=payload,
    )

    assert record.expert_id == "expert-a"
    assert record.location == ExpertLocation.RAM
    assert record.payload is payload
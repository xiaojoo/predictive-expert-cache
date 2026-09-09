from dataclasses import dataclass

from predictive_cache.storage.expert_store import ExpertLocation


@dataclass
class ExpertRecord:
    """Common data contract for a stored expert."""

    expert_id: str
    location: ExpertLocation
    payload: object
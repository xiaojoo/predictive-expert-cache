from dataclasses import dataclass

from predictive_cache.storage.expert_store import ExpertLocation
from predictive_cache.types import ExpertId


@dataclass
class ExpertRecord:
    """Common data contract for a stored expert."""

    expert_id: ExpertId
    location: ExpertLocation
    payload: object
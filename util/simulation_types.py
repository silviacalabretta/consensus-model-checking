from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class TrajectoryStep:
    time: float
    state: int
    labels: frozenset
    variables: Dict[str, int]


@dataclass
class EpisodeResult:
    steps: List[TrajectoryStep]
    final_time: float
    final_consensus_type: Optional[str]
    consensus_time_fraction: float
    consensus_entry_count: int
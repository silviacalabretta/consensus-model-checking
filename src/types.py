from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ModelParams:
    model_name: str
    N: int
    Za: int = 0
    Zb: int = 0
    C: int = 0
    t: int = 35
    h: int = 40
    qa: float = 1.05
    qb: float = 0.95

    def format_disruptors(self) -> str:
        if "zealot" in self.model_name:
            return f"Za={self.Za}, Zb={self.Zb}"
        return f"C={self.C}"

    def format_params(self) -> str:
        return (
            f"N={self.N}, {self.format_disruptors()}, "
            f"t={self.t}, h={self.h}, qa={self.qa}, qb={self.qb}"
        )


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


@dataclass
class BuiltModel:
    prism_program: object
    model: object


@dataclass
class SimulationContext:
    built_model: BuiltModel
    transitions: Dict[int, List[tuple]]
    exit_rates: List[float]
    var_names: List[str]
    var_values: Dict[str, list]
    initial_state: int

    @property
    def labeling(self):
        return self.built_model.model.labeling


@dataclass
class PropertyResult:
    name: str
    formula: str
    value: float
    elapsed_time: float


@dataclass
class ModelCheckResult:
    params: ModelParams
    build_time: float
    check_time: float
    nr_states: int
    nr_transitions: int
    initial_state: int
    initial_labels: list[str]
    probability_results: list[PropertyResult]
    expected_time_results: list[PropertyResult]

    def __str__(self) -> str:
        lines = [
            f"Model: {self.params.model_name}",
            f"Parameters: {self.params.format_params()}",
            (
                f"Built in {self.build_time:.3f}s — "
                f"{self.nr_states} states, {self.nr_transitions} transitions"
            ),
            f"Checked all properties in {self.check_time:.3f}s",
            "",
            f"Initial state: {self.initial_state}",
            f"Labels: {self.initial_labels}",
            "",
            "=" * 60,
            "PROBABILITY PROPERTIES",
            "=" * 60,
        ]

        for result in self.probability_results:
            lines.append(
                f"  {result.name}  =>  {result.value}  ({result.elapsed_time:.3f}s)"
            )

        lines.extend([
            "",
            "=" * 60,
            "EXPECTED TIME PROPERTIES",
            "=" * 60,
        ])

        for result in self.expected_time_results:
            lines.append(
                f"  {result.formula}  =>  {result.value}  ({result.elapsed_time:.3f}s)"
            )

        return "\n".join(lines)


@dataclass(frozen=True)
class Estimate:
    successes: int
    episodes: int
    probability: float
    standard_error: float

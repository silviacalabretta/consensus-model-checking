from __future__ import annotations

import random
from typing import Dict, List

from src.types import EpisodeResult, TrajectoryStep


def _compute_episode_stats(steps: List[TrajectoryStep], total_time: float) -> tuple:
    if total_time <= 0.0 or not steps:
        return None, 0

    consensus_time = 0.0
    entry_count = 0
    was_in_consensus = False

    for i, step in enumerate(steps):
        in_consensus = "consensus" in step.labels
        if in_consensus and not was_in_consensus:
            entry_count += 1

        interval_end = (
            steps[i + 1].time if i + 1 < len(steps)
            else total_time
        )

        dt = interval_end - step.time
        if dt < 0.0:
            raise ValueError("Trajectory times must be non-decreasing")

        if in_consensus:
            consensus_time += dt

        was_in_consensus = in_consensus

    return consensus_time / total_time, entry_count


def simulate_episode(
    transitions: Dict[int, List[tuple]],
    exit_rates: List[float],
    labeling,
    var_names: List[str],
    var_values: Dict[str, list],
    initial_state: int,
    max_time: float,
    rng: random.Random,
) -> EpisodeResult:
    state = initial_state
    steps: List[TrajectoryStep] = []
    current_time = 0.0

    def record_step() -> None:
        labels = frozenset(labeling.get_labels_of_state(state))
        variables = {
            name: int(var_values[name][state])
            for name in var_names
        }
        steps.append(
            TrajectoryStep(
                time=current_time,
                state=state,
                labels=labels,
                variables=variables,
            )
        )

    record_step()

    while current_time < max_time:
        rate = exit_rates[state]

        if rate <= 0.0:
            current_time = max_time
            break

        waiting_time = rng.expovariate(rate)
        next_time = current_time + waiting_time

        if next_time > max_time:
            current_time = max_time
            break

        threshold = rng.random() * rate
        cumulative = 0.0
        for target, t_rate in transitions[state]:
            if t_rate < 0.0:
                raise ValueError(f"Negative transition rate {t_rate} in state {state}")
            cumulative += t_rate
            if cumulative > threshold:
                state = target
                break
        else:
            raise RuntimeError(
                f"Failed to select a transition from state {state}. "
                "The exit rate may not equal the sum of outgoing rates."
            )

        current_time = next_time
        record_step()

    final_labels = frozenset(labeling.get_labels_of_state(state))
    final_consensus_type = None
    if "consensus" in final_labels:
        if "maj_a" in final_labels:
            final_consensus_type = "maj_a"
        elif "maj_b" in final_labels:
            final_consensus_type = "maj_b"

    frac, entries = _compute_episode_stats(steps, current_time)

    return EpisodeResult(
        steps=steps,
        final_time=current_time,
        final_consensus_type=final_consensus_type,
        consensus_time_fraction=frac if frac is not None else 0.0,
        consensus_entry_count=entries,
    )

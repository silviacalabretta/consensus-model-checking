from __future__ import annotations

import math
import random
from typing import List

from src.types import EpisodeResult, ModelParams, TrajectoryStep


def compute_initial_state(model_name: str, params: ModelParams) -> tuple[int, ...]:
    """Returns the initial population tuple for the model."""
    N = params.N
    Ahalf = N // 2
    Bhalf = N - Ahalf

    if model_name == "voter_zealots":
        return (Ahalf - params.Za, Bhalf - params.Zb)
    if model_name == "voter_contrarians":
        Cahalf = params.C // 2
        Cbhalf = params.C - Cahalf
        return (Ahalf - Cahalf, Bhalf - Cbhalf, Cahalf, Cbhalf)
    if model_name == "crossinh_zealots":
        return (Ahalf - params.Za, Bhalf - params.Zb, 0)
    if model_name == "crossinh_contrarians":
        Cahalf = params.C // 2
        Cbhalf = params.C - Cahalf
        return (Ahalf - Cahalf, Bhalf - Cbhalf, 0, Cahalf, Cbhalf)
    raise ValueError(f"Unknown model: {model_name}")


def compute_labels(model_name: str, state: tuple[int, ...], params: ModelParams) -> frozenset[str]:
    """Given a state tuple, returns which labels are satisfied."""
    N = params.N
    m = math.ceil(0.5 * N)          # majority
    d = max(1, math.ceil(N / 10))   # difference

    if "zealot" in model_name:
        supp_a = state[0] + params.Za
        supp_b = state[1] + params.Zb
    else:
        supp_a = state[0] + state[-2]
        supp_b = state[1] + state[-1]

    labels: set[str] = set()
    maj_a = supp_a >= m and supp_a >= supp_b + d
    maj_b = supp_b >= m and supp_b >= supp_a + d
    if maj_a:
        labels.add("maj_a")
    if maj_b:
        labels.add("maj_b")
    if maj_a or maj_b:
        labels.add("consensus")
    return frozenset(labels)


def compute_reactions(model_name: str, state: tuple[int, ...], params: ModelParams) -> list[tuple[tuple[int, ...], float]]:
    """Dispatches model-specific reaction function. Returns [(delta_tuple, rate), ...].    """
    ra = params.qa / params.N
    rb = params.qb / params.N

    if model_name == "voter_zealots":
        return _reactions_voter_zealots(state, ra, rb, params.Za, params.Zb)
    if model_name == "voter_contrarians":
        return _reactions_voter_contrarians(state, ra, rb, params.C)
    if model_name == "crossinh_zealots":
        return _reactions_crossinh_zealots(state, ra, rb, params.Za, params.Zb)
    if model_name == "crossinh_contrarians":
        return _reactions_crossinh_contrarians(state, ra, rb, params.C)
    raise ValueError(f"Unknown model: {model_name}")


def _reactions_voter_zealots(state: tuple[int, ...], ra: float, rb: float, Za: int, Zb: int) -> list[tuple[tuple[int, ...], float]]:
    a, b = state
    N = a + b + Za + Zb
    reactions: list[tuple[tuple[int, ...], float]] = []

    if a > 0 and b > 0 and a < N:
        reactions.append(((1, -1), ra * a * b))
    if a > 0 and b > 0 and b < N:
        reactions.append(((-1, 1), rb * a * b))
    if b > 0 and Za > 0 and a < N:
        reactions.append(((1, -1), ra * b * Za))
    if a > 0 and Zb > 0 and b < N:
        reactions.append(((-1, 1), rb * a * Zb))
    return reactions


def _reactions_voter_contrarians(state: tuple[int, ...], ra: float, rb: float, C: int) -> list[tuple[tuple[int, ...], float]]:
    a, b, Ca, Cb = state
    N = a + b + Ca + Cb
    reactions: list[tuple[tuple[int, ...], float]] = []

    if a > 0 and b > 0 and a < N:
        reactions.append(((1, -1, 0, 0), ra * a * b))
    if a > 0 and b > 0 and b < N:
        reactions.append(((-1, 1, 0, 0), rb * a * b))
    if a > 0 and Cb > 0 and b < N:
        reactions.append(((-1, 1, 0, 0), rb * a * Cb))
    if a > 0 and Ca > 0 and Cb < C:
        reactions.append(((0, 0, -1, 1), ra * a * Ca))
    if b > 0 and Ca > 0 and a < N:
        reactions.append(((1, -1, 0, 0), ra * b * Ca))
    if b > 0 and Cb > 0 and Ca < C:
        reactions.append(((0, 0, 1, -1), rb * b * Cb))
    if Ca > 1 and Cb < C - 1:
        reactions.append(((0, 0, -2, 2), ra * Ca * (Ca - 1) / 2))
    if Cb > 1 and Ca < C - 1:
        reactions.append(((0, 0, 2, -2), rb * Cb * (Cb - 1) / 2))
    return reactions


def _reactions_crossinh_zealots(state: tuple[int, ...], ra: float, rb: float, Za: int, Zb: int) -> list[tuple[tuple[int, ...], float]]:
    a, b, u = state
    N = a + b + u + Za + Zb
    reactions: list[tuple[tuple[int, ...], float]] = []

    if a > 0 and b > 0 and u < N:
        reactions.append(((0, -1, 1), ra * a * b))
    if a > 0 and b > 0 and u < N:
        reactions.append(((-1, 0, 1), rb * a * b))
    if a > 0 and u > 0 and a < N:
        reactions.append(((1, 0, -1), ra * a * u))
    if b > 0 and u > 0 and b < N:
        reactions.append(((0, 1, -1), rb * b * u))
    if a > 0 and Zb > 0 and u < N:
        reactions.append(((-1, 0, 1), rb * a * Zb))
    if u > 0 and Zb > 0 and b < N:
        reactions.append(((0, 1, -1), rb * u * Zb))
    if b > 0 and Za > 0 and u < N:
        reactions.append(((0, -1, 1), ra * b * Za))
    if u > 0 and Za > 0 and a < N:
        reactions.append(((1, 0, -1), ra * u * Za))
    return reactions


def _reactions_crossinh_contrarians(state: tuple[int, ...], ra: float, rb: float, C: int) -> list[tuple[tuple[int, ...], float]]:
    a, b, u, Ca, Cb = state
    N = a + b + u + Ca + Cb
    reactions: list[tuple[tuple[int, ...], float]] = []

    if a > 0 and b > 0 and u < N:
        reactions.append(((0, -1, 1, 0, 0), ra * a * b))
    if a > 0 and b > 0 and u < N:
        reactions.append(((-1, 0, 1, 0, 0), rb * a * b))
    if a > 0 and u > 0 and a < N:
        reactions.append(((1, 0, -1, 0, 0), ra * a * u))
    if b > 0 and u > 0 and b < N:
        reactions.append(((0, 1, -1, 0, 0), rb * b * u))
    if a > 0 and Cb > 0 and u < N:
        reactions.append(((-1, 0, 1, 0, 0), rb * a * Cb))
    if u > 0 and Cb > 0 and b < N:
        reactions.append(((0, 1, -1, 0, 0), rb * u * Cb))
    if a > 0 and Ca > 0 and Cb < C:
        reactions.append(((0, 0, 0, -1, 1), ra * a * Ca))
    if b > 0 and Ca > 0 and u < N:
        reactions.append(((0, -1, 1, 0, 0), ra * b * Ca))
    if u > 0 and Ca > 0 and a < N:
        reactions.append(((1, 0, -1, 0, 0), ra * u * Ca))
    if b > 0 and Cb > 0 and Ca < C:
        reactions.append(((0, 0, 0, 1, -1), rb * b * Cb))
    if Ca > 1 and Cb < C - 1:
        reactions.append(((0, 0, 0, -2, 2), ra * Ca * (Ca - 1) / 2))
    if Cb > 1 and Ca < C - 1:
        reactions.append(((0, 0, 0, 2, -2), rb * Cb * (Cb - 1) / 2))
    return reactions


def _step_simulation(
    state: tuple[int, ...],
    params: ModelParams,
    rng: random.Random,
    current_time: float,
    max_time: float,
) -> tuple[tuple[int, ...], float, float] | None:
    """ One Gillespie step from state. """
    reactions = compute_reactions(params.model_name, state, params)
    if not reactions:
        return None

    total_rate = sum(r for _, r in reactions)
    if total_rate <= 0.0:
        return None

    dt = rng.expovariate(total_rate)
    next_time = current_time + dt
    if next_time > max_time:
        return None

    threshold = rng.random() * total_rate
    cumulative = 0.0
    for delta, rate in reactions:
        cumulative += rate
        if cumulative > threshold:
            new_state = tuple(s + d for s, d in zip(state, delta))
            return new_state, next_time, dt

    last_delta, _ = reactions[-1]
    new_state = tuple(s + d for s, d in zip(state, last_delta))
    return new_state, next_time, dt


def simulate_direct_episode(
    params: ModelParams,
    rng: random.Random,
    max_time: float,
) -> EpisodeResult:
    state = compute_initial_state(params.model_name, params)
    current_time = 0.0
    steps: list[TrajectoryStep] = []

    def record() -> None:
        steps.append(
            TrajectoryStep(
                time=current_time,
                state=0,
                labels=compute_labels(params.model_name, state, params),
                variables={name: val for name, val in zip(_state_var_names(params.model_name), state)},
            )
        )

    record()

    while current_time < max_time:
        result = _step_simulation(state, params, rng, current_time, max_time)
        if result is None:
            current_time = max_time
            break
        state, current_time, _ = result
        record()

    final_labels = compute_labels(params.model_name, state, params)
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


def simulate_direct_with_monitors(
    params: ModelParams,
    rng: random.Random,
    max_time: float,
    stability_monitors: dict[str, tuple[str, float, float]],
    reach_monitors: dict[str, tuple[str, float]],
) -> dict[str, bool]:
    """Run one simulation, evaluate multiple monitors in a single pass.

    Args:
        stability_monitors: name -> (label, reach_bound, holding_time)
        reach_monitors: name -> (label, time_bound)

    Returns:
        dict mapping monitor name -> satisfied (bool)
    """
    state = compute_initial_state(params.model_name, params)
    current_time = 0.0

    sat_starts: dict[str, float | None] = {name: None for name in stability_monitors}
    results: dict[str, bool] = {}
    for name in stability_monitors:
        results[name] = False
    for name in reach_monitors:
        results[name] = False

    all_names = set(stability_monitors) | set(reach_monitors)

    def _check_stability_at(name: str, t: float) -> None:
        if results[name]:
            return
        start = sat_starts[name]
        if start is None:
            return
        _, reach_bound, holding_time = stability_monitors[name]
        if start <= reach_bound and t - start >= holding_time:
            results[name] = True

    def _check_state(labels: frozenset[str], t: float) -> None:
        for name, (label, _, _) in stability_monitors.items():
            if results[name]:
                continue
            if label in labels:
                if sat_starts[name] is None:
                    sat_starts[name] = t
            else:
                _check_stability_at(name, t)
                sat_starts[name] = None
        for name, (label, time_bound) in reach_monitors.items():
            if not results[name] and label in labels and t <= time_bound:
                results[name] = True

    initial_labels = compute_labels(params.model_name, state, params)
    _check_state(initial_labels, current_time)

    while current_time < max_time:
        if all(results[name] for name in all_names):
            break

        result = _step_simulation(state, params, rng, current_time, max_time)
        if result is None:
            current_time = max_time
            break
        state, current_time, _ = result

        labels = compute_labels(params.model_name, state, params)
        _check_state(labels, current_time)

    for name in stability_monitors:
        _check_stability_at(name, current_time)

    return results


def _state_var_names(model_name: str) -> list[str]:
    if model_name == "voter_zealots":
        return ["a", "b"]
    if model_name == "voter_contrarians":
        return ["a", "b", "Ca", "Cb"]
    if model_name == "crossinh_zealots":
        return ["a", "b", "u"]
    if model_name == "crossinh_contrarians":
        return ["a", "b", "u", "Ca", "Cb"]
    raise ValueError(f"Unknown model: {model_name}")


def _compute_episode_stats(
    steps: List[TrajectoryStep], total_time: float
) -> tuple[float | None, int]:
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
            steps[i + 1].time if i + 1 < len(steps) else total_time
        )
        dt = interval_end - step.time
        if dt < 0.0:
            raise ValueError("Trajectory times must be non-decreasing")
        if in_consensus:
            consensus_time += dt
        was_in_consensus = in_consensus

    return consensus_time / total_time, entry_count

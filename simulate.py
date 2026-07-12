#!/usr/bin/env python3
import random
import statistics
import time
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Optional

from util.cli_args import build_simulate_parser
from util.model_builder import (
    build_ctmc,
    get_exit_rates,
    get_initial_state,
    get_state_variables,
    get_transitions,
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


def _compute_episode_stats(steps: List[TrajectoryStep], total_time: float) -> tuple:
    if total_time <= 0.0 or not steps:
        return None, 0

    consensus_time = 0.0
    entry_count = 0
    was_in_consensus = False

    for i,step in enumerate(steps):
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

    return consensus_time/total_time, entry_count


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
            # deadlock/absorbing state
            current_time = max_time
            break

        waiting_time = rng.expovariate(rate)
        next_time = current_time + waiting_time

        if next_time > max_time:
            # no other transition before observation horizon
            current_time = max_time
            break
        

        # select transition j with probability rate_j / rate
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


def _print_results(
    results: List[EpisodeResult],
    var_names: List[str],
    max_time: float,
    elapsed_sim: float,
) -> None:
    n_episodes = len(results)

    if n_episodes == 0:
        raise ValueError("Cannot print results: no episodes were simulated")

    print(f"\n{'='*60}")
    print("SAMPLE TRAJECTORIES (first 3)")
    print("=" * 60)
    for idx, res in enumerate(results[:3]):
        var_header = " ".join(f"{n}" for n in var_names)
        print(f"\n--- Episode {idx+1} ({len(res.steps)} steps, "
              f"t={res.final_time:.4f}, "
              f"consensus_frac={res.consensus_time_fraction:.4f}, "
              f"entries={res.consensus_entry_count}, "
              f"final={res.final_consensus_type or 'none'}) ---")
        print(f"  {'t':>10s}  {var_header}  labels")
        sample_every = max(1, len(res.steps) // 20)
        for i in range(0, len(res.steps), sample_every):
            s = res.steps[i]
            var_str = " ".join(f"{s.variables[n]}" for n in var_names)
            label_str = ",".join(sorted(s.labels)) if s.labels else "-"
            print(f"  {s.time:10.4f}  {var_str}  [{label_str}]")
        last = res.steps[-1]
        if (len(res.steps) - 1) % sample_every != 0:
            var_str = " ".join(f"{last.variables[n]}" for n in var_names)
            label_str = ",".join(sorted(last.labels)) if last.labels else "-"
            print(f"  {last.time:10.4f}  {var_str}  [{label_str}]  (final)")

    fractions = [r.consensus_time_fraction for r in results]
    entries = [r.consensus_entry_count for r in results]
    a_count = sum(1 for r in results if r.final_consensus_type == "maj_a")
    b_count = sum(1 for r in results if r.final_consensus_type == "maj_b")
    no_final = sum(1 for r in results if r.final_consensus_type is None)

    print(f"\n{'='*60}")
    print("AGGREGATED STATISTICS")
    print("=" * 60)
    print(f"  Episodes:              {n_episodes}")
    print(f"  Max simulation time:   {max_time}")

    print("\n  Time-weighted consensus fraction:")
    print(f"    Mean:   {statistics.mean(fractions):.4f}")
    print(f"    Std:    {statistics.stdev(fractions) if len(fractions) > 1 else 0:.4f}")
    print(f"    Median: {statistics.median(fractions):.4f}")
    print(f"    Min:    {min(fractions):.4f}")
    print(f"    Max:    {max(fractions):.4f}")

    print("\n  Consensus entry count:")
    print(f"    Mean:   {statistics.mean(entries):.4f}")
    print(f"    Std:    {statistics.stdev(entries) if len(entries) > 1 else 0:.4f}")
    print(f"    Median: {statistics.median(entries):.4f}")

    print(f"\n  Final state at t={max_time}:")
    print(
        f"    A-majority:   {a_count} "
        f"({a_count / n_episodes:.4f})"
    )
    print(
        f"    B-majority:   {b_count} "
        f"({b_count / n_episodes:.4f})"
    )
    print(
        f"    No consensus: {no_final} "
        f"({no_final / n_episodes:.4f})"
    )
    # print(f"    A-majority:  {a_count} ({a_count/episodes:.4f})")
    # print(f"    B-majority:  {b_count} ({b_count/episodes:.4f})")
    # print(f"    No consensus: {no_final} ({no_final/episodes:.4f})")

    print(f"\n  Total simulation time: {elapsed_sim:.3f}s")


def _validate_simulation_params(t: int, max_time: float, episodes: int) -> None:
    if max_time < t:
        raise ValueError(f"max_time ({max_time}) must be >= t ({t})")
    if episodes <= 0:
        raise ValueError("episodes must be positive")


def run_simulation(
    model_name: str,
    N: int, Za: int, Zb: int, C: int,
    t: int, h: int,
    n_episodes: int,  max_time: Optional[float], seed: int,
) -> None:
    if max_time is None:
        max_time = float(t)

    _validate_simulation_params(t, max_time, n_episodes)

    print(f"\nBuilding CTMC: {model_name}")
    print(f"  N={N}, Za={Za}, Zb={Zb}, C={C}, t={t}, h={h}")
    print(f"  Max simulation time: {max_time}")
    t0 = time.perf_counter()
    bm = build_ctmc(model_name, N=N, Za=Za, Zb=Zb, C=C, t=t, h=h)
    elapsed = time.perf_counter() - t0
    print(f"  Built in {elapsed:.3f}s — {bm.model.nr_states} states, "
          f"{bm.model.nr_transitions} transitions")

    transitions = get_transitions(bm.model)
    exit_rates = get_exit_rates(bm.model)
    var_names, var_values = get_state_variables(bm.model, bm.prism_program, model_name)
    initial_state = get_initial_state(bm.model)

    rng = random.Random(seed)
    print(f"\nRunning {n_episodes} episodes (seed={seed})...")
    t0 = time.perf_counter()

    results: List[EpisodeResult] = []
    # for ep in range(episodes):
    for _ in tqdm(range(n_episodes), desc="Simulating episodes", unit="episode"):
        results.append(simulate_episode(
            transitions, exit_rates, bm.model.labeling,
            var_names, var_values, initial_state,
            max_time, rng,
        ))
        # if (ep + 1) % max(1, episodes // 10) == 0:
        #     print(f"  {ep+1}/{episodes} done")

    elapsed_sim = time.perf_counter() - t0
    print(f"  Done in {elapsed_sim:.3f}s")

    _print_results(results, var_names, max_time, elapsed_sim)


def main():
    args = build_simulate_parser().parse_args()
    run_simulation(
        args.model, args.N, args.Za, args.Zb, args.C,
        args.t, args.h, args.episodes,
        args.max_time, args.seed,
    )


if __name__ == "__main__":
    main()

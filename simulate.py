#!/usr/bin/env python3

import random
import statistics
import time
from typing import List

from tqdm import tqdm

from src.cli import build_simulate_parser, resolve_model_args
from src.gillespie_storm import simulate_episode
from src.model import build_simulation_context
from src.plotting import plot_trajectories
from src.types import EpisodeResult, ModelParams


def _validate_simulation_params(t: int, h: int, max_time: float, episodes: int) -> None:
    if max_time < t + h:
        raise ValueError(f"max_time ({max_time}) must be >= t+h ({t + h})")
    if episodes <= 0:
        raise ValueError("episodes must be positive")


def _print_results(
    results: List[EpisodeResult],
    var_names: List[str],
    max_time: float,
    elapsed_sim: float,
) -> None:
    n_episodes = len(results)

    if n_episodes == 0:
        raise ValueError("Cannot print results: no episodes were simulated")

    print(f"\n{'=' * 60}")
    print("SAMPLE TRAJECTORIES (first 3)")
    print("=" * 60)
    for idx, res in enumerate(results[:3]):
        var_header = " ".join(f"{n}" for n in var_names)
        print(
            f"\n--- Episode {idx + 1} ({len(res.steps)} steps, "
            f"t={res.final_time:.4f}, "
            f"consensus_frac={res.consensus_time_fraction:.4f}, "
            f"entries={res.consensus_entry_count}, "
            f"final={res.final_consensus_type or 'none'}) ---"
        )
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
            print(f"  {last.time:10.4f}  {var_str}  [{label_str}] ")

    fractions = [r.consensus_time_fraction for r in results]
    entries = [r.consensus_entry_count for r in results]
    a_count = sum(1 for r in results if r.final_consensus_type == "maj_a")
    b_count = sum(1 for r in results if r.final_consensus_type == "maj_b")
    no_final = sum(1 for r in results if r.final_consensus_type is None)

    print(f"\n{'=' * 60}")
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
    print(f"    A-majority:   {a_count} ({a_count / n_episodes:.4f})")
    print(f"    B-majority:   {b_count} ({b_count / n_episodes:.4f})")
    print(f"    No consensus: {no_final} ({no_final / n_episodes:.4f})")

    print(f"\n  Total simulation time: {elapsed_sim:.3f}s")


def run_simulation(
    params: ModelParams,
    n_episodes: int,
    max_time: float,
    seed: int,
) -> None:
    _validate_simulation_params(params.t, params.h, max_time, n_episodes)

    print(f"\nBuilding CTMC: {params.model_name}")
    print(f"  {params.format_params()}")
    print(f"  Max simulation time: {max_time}")
    t0 = time.perf_counter()
    ctx = build_simulation_context(params)
    elapsed = time.perf_counter() - t0
    model = ctx.built_model.model
    print(
        f"  Built in {elapsed:.3f}s — {model.nr_states} states, "
        f"{model.nr_transitions} transitions"
    )

    rng = random.Random(seed)
    print(f"\nRunning {n_episodes} episodes (seed={seed})...")
    t0 = time.perf_counter()

    results: List[EpisodeResult] = []
    for _ in tqdm(range(n_episodes), desc="Simulating episodes", unit="episode"):
        results.append(
            simulate_episode(
                ctx.transitions,
                ctx.exit_rates,
                ctx.labeling,
                ctx.var_names,
                ctx.var_values,
                ctx.initial_state,
                max_time,
                rng,
            )
        )

    elapsed_sim = time.perf_counter() - t0
    print(f"  Done in {elapsed_sim:.3f}s")

    _print_results(results, ctx.var_names, max_time, elapsed_sim)


def main() -> None:
    args = build_simulate_parser().parse_args()
    args = resolve_model_args(args)
    params = ModelParams(
        model_name=args.model,
        N=args.N,
        Za=args.Za,
        Zb=args.Zb,
        C=args.C,
        t=args.t,
        h=args.h,
        qa=args.qa,
        qb=args.qb,
    )
    run_simulation(
        params,
        args.episodes,
        args.max_time,
        args.seed,
    )


if __name__ == "__main__":
    main()

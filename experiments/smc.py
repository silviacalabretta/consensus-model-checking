#!/usr/bin/env python3

import argparse
import math
import random
import time

from tqdm import tqdm

from src.cli import build_experiment_parser, resolve_model_args
from src.estimation import wilson_interval
from src.gillespie import simulate_episode
from src.model import build_simulation_context
from src.monitors import reaches_and_holds
from src.types import ModelParams


STABILITY_LABELS = {
    "stable_a": "maj_a",
    "stable_b": "maj_b",
}


def estimate_stable_consensus(
    params: ModelParams,
    episodes: int,
    seed: int,
    show_progress: bool,
) -> tuple[dict[str, int], float]:
    """Estimate stable-consensus properties through simulation."""
    ctx = build_simulation_context(params)
    model = ctx.built_model.model

    successes = {
        "stable_a": 0,
        "stable_b": 0,
        "stable_consensus": 0,
    }

    rng = random.Random(seed)

    iterator = range(episodes)
    if show_progress:
        iterator = tqdm(iterator, desc="Simulating", unit="episode")

    start = time.perf_counter()

    for _ in iterator:
        episode = simulate_episode(
            transitions=ctx.transitions,
            exit_rates=ctx.exit_rates,
            labeling=model.labeling,
            var_names=ctx.var_names,
            var_values=ctx.var_values,
            initial_state=ctx.initial_state,
            max_time=params.t + params.h,
            rng=rng,
        )

        outcomes = {
            property_name: reaches_and_holds(
                episode, label=label,
                reach_bound=params.t, holding_time=params.h,
            )
            for property_name, label in STABILITY_LABELS.items()
        }

        for property_name, satisfied in outcomes.items():
            if satisfied:
                successes[property_name] += 1

        if outcomes["stable_a"] or outcomes["stable_b"]:
            successes["stable_consensus"] += 1

    elapsed = time.perf_counter() - start
    return successes, elapsed


def print_estimates(
    successes: dict[str, int],
    episodes: int,
    confidence: float,
) -> None:
    print()
    print(
        f"{'Property':<22}"
        f"{'Successes':>12}"
        f"{'Estimate':>12}"
        f"{'Std. error':>14}"
        f"{'CI lower':>12}"
        f"{'CI upper':>12}"
    )
    print("-" * 84)

    for property_name, count in successes.items():
        estimate = count / episodes
        standard_error = math.sqrt(estimate * (1.0 - estimate) / episodes)
        lower, upper = wilson_interval(
            successes=count, episodes=episodes, confidence=confidence,
        )
        print(
            f"{property_name:<22}"
            f"{count:>12}"
            f"{estimate:>12.6f}"
            f"{standard_error:>14.6f}"
            f"{lower:>12.6f}"
            f"{upper:>12.6f}"
        )


def run_smc(args: argparse.Namespace) -> None:
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    params = ModelParams(
        model_name=args.model,
        N=args.N, Za=args.Za, Zb=args.Zb,
        C=args.C, t=args.t, h=args.h,
        qa=args.qa, qb=args.qb,
    )

    print("Configuration")
    print(f"  model: {params.model_name}")
    print(f"  {params.format_params()}")
    print(f"  simulation horizon={params.t + params.h}")
    print(f"  episodes={args.episodes}, seed={args.seed}")
    print(f"  confidence={args.confidence:.3f}")

    successes, simulation_time = estimate_stable_consensus(
        params=params,
        episodes=args.episodes,
        seed=args.seed,
        show_progress=not args.no_progress,
    )

    print_estimates(
        successes=successes,
        episodes=args.episodes,
        confidence=args.confidence,
    )

    print(
        f"\nSimulation time: {simulation_time:.3f}s "
        f"({simulation_time / args.episodes:.6f}s per episode)"
    )


def main() -> None:
    parser = build_experiment_parser(
        description=(
            "Estimate stable-consensus probabilities using "
            "Statistical Model Checking."
        ),
        default_episodes=4500,
    )
    args = parser.parse_args()
    args = resolve_model_args(args)
    run_smc(args)


if __name__ == "__main__":
    main()

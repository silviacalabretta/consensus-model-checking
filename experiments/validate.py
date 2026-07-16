#!/usr/bin/env python3

import argparse
import random
import time
from typing import Dict

from tqdm import tqdm

from src.checking import run_model_check
from src.cli import build_experiment_parser, resolve_model_args
from src.estimation import wilson_interval
from src.gillespie import simulate_episode
from src.model import build_simulation_context
from src.monitors import reaches_label_before
from src.types import ModelParams


PROPERTY_LABELS = {
    "reach_a": "maj_a",
    "reach_b": "maj_b",
    "reach_consensus": "consensus",
}


def exact_probabilities(model_check_result) -> Dict[str, float]:
    """Extract the exact reachability probabilities by property name."""
    values = {
        result.name: float(result.value)
        for result in model_check_result.probability_results
    }

    missing = set(PROPERTY_LABELS) - set(values)
    if missing:
        raise ValueError(f"Missing exact properties: {sorted(missing)}")

    return {
        property_name: values[property_name]
        for property_name in PROPERTY_LABELS
    }


def estimate_probabilities(
    params: ModelParams,
    episodes: int,
    seed: int,
    show_progress: bool,
) -> tuple[Dict[str, int], float]:
    """Simulate episodes and count bounded-reachability successes."""
    ctx = build_simulation_context(params)
    model = ctx.built_model.model

    successes = {property_name: 0 for property_name in PROPERTY_LABELS}
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
            max_time=params.t,
            rng=rng,
        )

        for property_name, label in PROPERTY_LABELS.items():
            if reaches_label_before(episode, label, params.t):
                successes[property_name] += 1

    elapsed = time.perf_counter() - start
    return successes, elapsed


def print_comparison(
    exact: Dict[str, float],
    successes: Dict[str, int],
    episodes: int,
    confidence: float,
) -> None:
    print()
    print(
        f"{'Property':<20}"
        f"{'Exact':>12}"
        f"{'Estimate':>12}"
        f"{'CI lower':>12}"
        f"{'CI upper':>12}"
        f"{'Abs. error':>14}"
        f"{'Exact in CI':>14}"
    )
    print("-" * 96)

    for property_name in PROPERTY_LABELS:
        exact_value = exact[property_name]
        estimate = successes[property_name] / episodes
        lower, upper = wilson_interval(
            successes[property_name], episodes, confidence,
        )
        absolute_error = abs(estimate - exact_value)
        tol = 1e-12
        inside = lower - tol <= exact_value <= upper + tol

        print(
            f"{property_name:<20}"
            f"{exact_value:>12.6f}"
            f"{estimate:>12.6f}"
            f"{lower:>12.6f}"
            f"{upper:>12.6f}"
            f"{absolute_error:>14.6f}"
            f"{str(inside):>14}"
        )


def run_validation(args: argparse.Namespace) -> None:
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

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

    print("Configuration")
    print(f"  model: {params.model_name}")
    print(f"  {params.format_params()}")
    print(f"  episodes={args.episodes}, seed={args.seed}")
    print(f"  confidence={args.confidence:.3f}")

    print("\nRunning exact model checking...")
    exact_result = run_model_check(params)
    exact = exact_probabilities(exact_result)

    print(
        f"  {exact_result.nr_states} states, "
        f"{exact_result.nr_transitions} transitions"
    )
    print(
        f"  build={exact_result.build_time:.3f}s, "
        f"check={exact_result.check_time:.3f}s"
    )

    print("\nRunning Gillespie simulations...")
    successes, simulation_time = estimate_probabilities(
        params=params,
        episodes=args.episodes,
        seed=args.seed,
        show_progress=not args.no_progress,
    )

    print_comparison(
        exact=exact,
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
            "Validate Gillespie simulation against exact bounded "
            "reachability probabilities."
        ),
        default_episodes=1000,
    )
    args = parser.parse_args()
    args = resolve_model_args(args)
    run_validation(args)


if __name__ == "__main__":
    main()

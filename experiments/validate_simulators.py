#!/usr/bin/env python3

import random
import time

from tqdm import tqdm

from src.checking import check_properties
from src.cli import build_experiment_parser, resolve_model_args
from src.estimation import wilson_interval
from src.gillespie_py import simulate_direct_episode, simulate_direct_with_monitors
from src.gillespie_storm import simulate_episode
from src.model import build_simulation_context
from src.monitors import reaches_and_holds, reaches_label_before
from src.types import EpisodeResult, ModelParams


REACH_LABELS = {
    "reach_a": "maj_a",
    "reach_b": "maj_b",
    "reach_consensus": "consensus",
}

STABLE_LABELS = {
    "stable_a": "maj_a",
    "stable_b": "maj_b",
}

PROPERTY_NAMES = tuple(REACH_LABELS) + tuple(STABLE_LABELS)


def build_monitors(params: ModelParams):
    t = float(params.t)
    h = float(params.h)
    stability = {
        name: (label, t, h)
        for name, label in STABLE_LABELS.items()
    }

    reach = {
        name: (label, t)
        for name, label in REACH_LABELS.items()
    }

    return stability, reach


def episode_outcomes(
    episode: EpisodeResult,
    params: ModelParams,
) -> dict[str, bool]:
    outcomes = {
        name: reaches_label_before(
            episode,
            label=label,
            time_bound=params.t,
        )
        for name, label in REACH_LABELS.items()
    }

    outcomes.update({
        name: reaches_and_holds(
            episode,
            label=label,
            reach_bound=params.t,
            holding_time=params.h,
        )
        for name, label in STABLE_LABELS.items()
    })

    return outcomes


def exact_reach_probabilities(ctx) -> dict[str, float]:
    """Model-check exact bounded-reachability properties."""
    probability_results = check_properties(
        property_type="prob",
        model=ctx.built_model.model,
        prism_program=ctx.built_model.prism_program,
        initial_state=ctx.initial_state,
    )

    values = {
        result.name: float(result.value)
        for result in probability_results
    }

    missing = set(REACH_LABELS) - set(values)
    if missing:
        raise ValueError(
            f"Missing exact properties: {sorted(missing)}"
        )

    return {
        name: values[name]
        for name in REACH_LABELS
    }


def add_outcomes(
    successes: dict[str, int],
    outcomes: dict[str, bool],
) -> None:
    for name, satisfied in outcomes.items():
        successes[name] += int(satisfied)


def format_interval(
    successes: int,
    episodes: int,
    confidence: float,
) -> tuple[float, float, float]:
    estimate = successes / episodes
    lower, upper = wilson_interval(
        successes,
        episodes,
        confidence,
    )
    return estimate, lower, upper


def print_online_agreement(
    agreements: dict[str, int],
    episodes: int,
) -> None:
    print("\nOnline vs offline direct Python monitors")
    print(f"{'Property':<22}{'Agreement':>15}{'Rate':>12}")
    print("-" * 49)

    for name in PROPERTY_NAMES:
        count = agreements[name]
        print(
            f"{name:<22}"
            f"{count:>9}/{episodes}"
            f"{count / episodes:>12.4%}"
        )

    if all(count == episodes for count in agreements.values()):
        print("\nPASS: online and offline direct Python monitors agree.")
    else:
        print("\nFAIL: online and offline direct Python monitors disagree.")


def print_exact_reachability(
    exact: dict[str, float],
    storm_successes: dict[str, int],
    direct_successes: dict[str, int],
    episodes: int,
    confidence: float,
) -> None:
    print("\nExact model checking vs simulation")
    print(
        f"{'Property':<18}"
        f"{'Exact':>10}"
        f"{'Storm sim.':>10}"
        f"{'Storm CI':>23}"
        f"{'In CI':>8}"
        f"{'Python sim.':>10}"
        f"{'Python CI':>23}"
        f"{'In CI':>8}"
    )
    print("-" * 115)

    tolerance = 1e-12

    for name in REACH_LABELS:
        exact_value = exact[name]

        storm_est, storm_lo, storm_hi = format_interval(
            storm_successes[name],
            episodes,
            confidence,
        )
        direct_est, direct_lo, direct_hi = format_interval(
            direct_successes[name],
            episodes,
            confidence,
        )

        storm_inside = (
            storm_lo - tolerance
            <= exact_value
            <= storm_hi + tolerance
        )
        direct_inside = (
            direct_lo - tolerance
            <= exact_value
            <= direct_hi + tolerance
        )

        storm_ci = f"[{storm_lo:.4f}, {storm_hi:.4f}]"
        direct_ci = f"[{direct_lo:.4f}, {direct_hi:.4f}]"

        print(
            f"{name:<18}"
            f"{exact_value:>10.6f}"
            f"{storm_est:>12.6f}"
            f"{storm_ci:>23}"
            f"{str(storm_inside):>8}"
            f"{direct_est:>13.6f}"
            f"{direct_ci:>23}"
            f"{str(direct_inside):>8}"
        )


def print_stability_comparison(
    storm_successes: dict[str, int],
    direct_successes: dict[str, int],
    episodes: int,
    confidence: float,
) -> None:
    print("\nStorm-based vs direct Python stability estimates")
    print(
        f"{'Property':<18}"
        f"{'Storm sim.':>10}"
        f"{'Storm CI':>23}"
        f"{'Python sim.':>10}"
        f"{'Python CI':>23}"
        f"{'Abs. diff':>12}"
        f"{'CI overlap':>12}"
    )
    print("-" * 113)

    for name in STABLE_LABELS:
        storm_est, storm_lo, storm_hi = format_interval(
            storm_successes[name],
            episodes,
            confidence,
        )
        direct_est, direct_lo, direct_hi = format_interval(
            direct_successes[name],
            episodes,
            confidence,
        )

        overlap = max(storm_lo, direct_lo) <= min(storm_hi, direct_hi)

        storm_ci = f"[{storm_lo:.4f}, {storm_hi:.4f}]"
        direct_ci = f"[{direct_lo:.4f}, {direct_hi:.4f}]"

        print(
            f"{name:<18}"
            f"{storm_est:>12.6f}"
            f"{storm_ci:>23}"
            f"{direct_est:>13.6f}"
            f"{direct_ci:>23}"
            f"{abs(storm_est - direct_est):>12.6f}"
            f"{str(overlap):>12}"
        )


def main() -> None:
    parser = build_experiment_parser(
        description=(
            "Validate the direct Gillespie simulator and its "
            "online monitors."
        ),
        default_episodes=1000,
    )

    args = resolve_model_args(parser.parse_args())

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

    max_time = params.t + params.h
    stability_monitors, reach_monitors = build_monitors(params)

    print("Configuration")
    print(f"  model: {params.model_name}")
    print(f"  {params.format_params()}")
    print(f"  horizon={max_time}")
    print(f"  episodes={args.episodes}, seed={args.seed}")
    print(f"  confidence={args.confidence:.3f}")

    print("\nBuilding CTMC with Stormpy...")
    ctx = build_simulation_context(params)
    model = ctx.built_model.model
    print(
        f"  {model.nr_states} states, "
        f"{model.nr_transitions} transitions"
    )

    print("\nComputing exact reachability probabilities with Storm...")
    exact = exact_reach_probabilities(ctx)

    storm_successes = {
        name: 0
        for name in PROPERTY_NAMES
    }
    direct_successes = {
        name: 0
        for name in PROPERTY_NAMES
    }
    online_successes = {
        name: 0
        for name in PROPERTY_NAMES
    }
    online_agreements = {
        name: 0
        for name in PROPERTY_NAMES
    }

    # Generates independent episode seeds.
    seed_generator = random.Random(args.seed)

    iterator = range(args.episodes)
    if not args.no_progress:
        iterator = tqdm(
            iterator,
            desc="Validating",
            unit="episode",
        )

    start = time.perf_counter()

    for _ in iterator:
        storm_seed = seed_generator.getrandbits(64)
        direct_seed = seed_generator.getrandbits(64)

        # Storm and direct are independent samples.
        storm_episode = simulate_episode(
            transitions=ctx.transitions,
            exit_rates=ctx.exit_rates,
            labeling=ctx.labeling,
            var_names=ctx.var_names,
            var_values=ctx.var_values,
            initial_state=ctx.initial_state,
            max_time=max_time,
            rng=random.Random(storm_seed),
        )

        # These two receive identical per-episode random streams.
        direct_episode = simulate_direct_episode(
            params=params,
            rng=random.Random(direct_seed),
            max_time=max_time,
        )

        online_outcomes = simulate_direct_with_monitors(
            params=params,
            rng=random.Random(direct_seed),
            max_time=max_time,
            stability_monitors=stability_monitors,
            reach_monitors=reach_monitors,
        )

        storm_outcomes = episode_outcomes(
            storm_episode,
            params,
        )
        direct_outcomes = episode_outcomes(
            direct_episode,
            params,
        )

        add_outcomes(storm_successes, storm_outcomes)
        add_outcomes(direct_successes, direct_outcomes)
        add_outcomes(online_successes, online_outcomes)

        for name in PROPERTY_NAMES:
            online_agreements[name] += int(
                online_outcomes[name] == direct_outcomes[name]
            )

    elapsed = time.perf_counter() - start

    print_online_agreement(
        online_agreements,
        args.episodes,
    )

    print_exact_reachability(
        exact=exact,
        storm_successes=storm_successes,
        direct_successes=direct_successes,
        episodes=args.episodes,
        confidence=args.confidence,
    )

    print_stability_comparison(
        storm_successes=storm_successes,
        direct_successes=direct_successes,
        episodes=args.episodes,
        confidence=args.confidence,
    )

    print(
        f"\nValidation time: {elapsed:.3f}s "
        f"({elapsed / args.episodes:.6f}s per episode triplet)"
    )


if __name__ == "__main__":
    main()
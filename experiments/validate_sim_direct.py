#!/usr/bin/env python3

import random
import time

from tqdm import tqdm

from src.cli import build_experiment_parser, resolve_model_args
from src.gillespie_py import simulate_direct_episode, simulate_direct_with_monitors
from src.gillespie_storm import simulate_episode
from src.model import build_simulation_context
from src.monitors import reaches_and_holds, reaches_label_before
from src.types import ModelParams


REACH_LABELS = {
    "reach_a": "maj_a",
    "reach_b": "maj_b",
    "reach_consensus": "consensus",
}

STABLE_LABELS = {
    "stable_a": "maj_a",
    "stable_b": "maj_b",
}

def _build_monitors(params: ModelParams):
    t = float(params.t)
    h = float(params.h)
    stability = {
        "stable_a": ("maj_a", t, h),
        "stable_b": ("maj_b", t, h),
    }
    reach = {
        "reach_a": ("maj_a", t),
        "reach_b": ("maj_b", t),
        "reach_consensus": ("consensus", t),
    }
    return stability, reach


def _count_agree(expected: dict[str, bool], actual: dict[str, bool]) -> dict[str, int]:
    return {
        name: int(expected[name] == actual[name])
        for name in expected
    }


def _print_table(title: str, rows: list[tuple[str, int, int]], total: int) -> None:
    print()
    print(title)
    print(f"{'Property':<25} {'Agree':>10} {'Rate':>10}")
    print("-" * 47)
    for name, agree, _ in rows:
        print(f"  {name:<23} {agree:>8}/{total} {agree/total:>9.4%}")
    all_pass = all(agree == total for _, agree, _ in rows)
    print()
    if all_pass:
        print("PASS: 100% agreement.")
    else:
        print("WARN: Disagreements found.")


def main() -> None:
    parser = build_experiment_parser(
        description=(
            "Validate simulators: Storm-based, direct episode, "
            "and direct with inline monitors."
        ),
        default_episodes=1000,
    )
    args = parser.parse_args()
    args = resolve_model_args(args)

    if args.episodes <= 0:
        raise ValueError("episodes must be positive")

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
    stability_monitors, reach_monitors = _build_monitors(params)
    all_monitor_names = sorted(set(stability_monitors) | set(reach_monitors))

    print("Configuration")
    print(f"  model: {params.model_name}")
    print(f"  {params.format_params()}")
    print(f"  episodes={args.episodes}, seed={args.seed}")
    print()

    print("Building Storm CTMC for reference simulator...")
    ctx = build_simulation_context(params)
    model = ctx.built_model.model
    nr_states = model.nr_states if hasattr(model, "nr_states") else "?"
    print(f"  {nr_states} states")
    print()

    rng_storm = random.Random(args.seed)
    rng_direct = random.Random(args.seed)
    rng_monitors = random.Random(args.seed)

    ep_reach_agree = {name: 0 for name in REACH_LABELS}
    ep_stable_agree = {name: 0 for name in STABLE_LABELS}
    mon_storm_agree = {name: 0 for name in all_monitor_names}
    mon_direct_agree = {name: 0 for name in all_monitor_names}

    total = args.episodes

    iterator = range(total)
    if not args.no_progress:
        iterator = tqdm(iterator, desc="Validating", unit="episode")

    t0 = time.perf_counter()

    for _ in iterator:
        ep_storm = simulate_episode(
            transitions=ctx.transitions,
            exit_rates=ctx.exit_rates,
            labeling=model.labeling,
            var_names=ctx.var_names,
            var_values=ctx.var_values,
            initial_state=ctx.initial_state,
            max_time=max_time,
            rng=rng_storm,
        )

        ep_direct = simulate_direct_episode(
            params=params,
            rng=rng_direct,
            max_time=max_time,
        )

        inline = simulate_direct_with_monitors(
            params=params,
            rng=rng_monitors,
            max_time=max_time,
            stability_monitors=stability_monitors,
            reach_monitors=reach_monitors,
        )

        for prop_name, label in REACH_LABELS.items():
            if reaches_label_before(ep_storm, label, params.t) == reaches_label_before(ep_direct, label, params.t):
                ep_reach_agree[prop_name] += 1

        for prop_name, label in STABLE_LABELS.items():
            if reaches_and_holds(ep_storm, label, params.t, params.h) == reaches_and_holds(ep_direct, label, params.t, params.h):
                ep_stable_agree[prop_name] += 1

        storm_mon = {}
        for name, label in stability_monitors.items():
            storm_mon[name] = reaches_and_holds(ep_storm, label[0], label[1], label[2])
        for name, label in reach_monitors.items():
            storm_mon[name] = reaches_label_before(ep_storm, label[0], label[1])

        direct_mon = {}
        for name, label in stability_monitors.items():
            direct_mon[name] = reaches_and_holds(ep_direct, label[0], label[1], label[2])
        for name, label in reach_monitors.items():
            direct_mon[name] = reaches_label_before(ep_direct, label[0], label[1])

        storm_counts = _count_agree(storm_mon, inline)
        direct_counts = _count_agree(direct_mon, inline)
        for name in all_monitor_names:
            mon_storm_agree[name] += storm_counts[name]
            mon_direct_agree[name] += direct_counts[name]

    elapsed = time.perf_counter() - t0

    ep_rows = (
        [(name, ep_reach_agree[name], total) for name in REACH_LABELS]
        + [(name, ep_stable_agree[name], total) for name in STABLE_LABELS]
    )
    _print_table("Episode-level: storm vs direct", ep_rows, total)

    mon_storm_rows = [(name, mon_storm_agree[name], total) for name in all_monitor_names]
    _print_table("Inline monitors vs storm episode monitors", mon_storm_rows, total)

    mon_direct_rows = [(name, mon_direct_agree[name], total) for name in all_monitor_names]
    _print_table("Inline monitors vs direct episode monitors", mon_direct_rows, total)

    print(
        f"\nValidation time: {elapsed:.3f}s "
        f"({elapsed / total:.6f}s per episode triplet)"
    )


if __name__ == "__main__":
    main()

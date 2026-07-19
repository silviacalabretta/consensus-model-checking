#!/usr/bin/env python3

import csv
import hashlib
import random
import time
from pathlib import Path

from tqdm import tqdm

from src.cli import build_sweep_parser
from src.estimation import wilson_interval
from src.gillespie_py import simulate_direct_with_monitors

from grid_common import GridConfig, generate_configs


T = 35

H_VALUES = [0, 5, 10, 20, 30, 40, 60, 80]
N_VALUES = [100, 1000]

QA = 1.05
QB = 0.95


# ============================================================
# TODO: SET THE DISRUPTOR FRACTIONS HERE
#
# Suggested structure for each model:
#   0.0,
#   one fraction below the transition,
#   one fraction near the transition,
#   one fraction above the transition.
# ============================================================

D_FRACTIONS_BY_MODEL: dict[str, list[float]] = {
    "voter_zealots": [0.00, 0.06, 0.10, 0.16, 0.20, 0.25, 0.3, 0.35, 0.4],
    "voter_contrarians": [0.00, 0.05, 0.08, 0.10, 0.16, 0.2, 0.25, 0.3, 0.35],
    "crossinh_zealots": [0.00, 0.20, 0.30, 0.36, 0.46, 0.5, 0.55],
    "crossinh_contrarians": [0.00, 0.10, 0.16, 0.20, 0.26, 0.3, 0.35],
}


MODEL_ORDER = (
    "voter_zealots",
    "voter_contrarians",
    "crossinh_zealots",
    "crossinh_contrarians",
)


CSV_FIELDS = (
    "model",
    "N",
    "Za",
    "Zb",
    "C",
    "disruptor_count",
    "disruptor_fraction",
    "t",
    "h",
    "qa",
    "qb",
    "simulation_horizon",

    "stable_a_successes",
    "stable_a",
    "stable_a_ci_lo",
    "stable_a_ci_hi",

    "stable_b_successes",
    "stable_b",
    "stable_b_ci_lo",
    "stable_b_ci_hi",

    "stable_consensus_successes",
    "stable_consensus",
    "stable_consensus_ci_lo",
    "stable_consensus_ci_hi",

    "episodes",
    "confidence",
    "base_seed",
    "config_seed",
    "total_time",
    "status",
    "error",
)


BASE_KEY_FIELDS = (
    "model",
    "N",
    "Za",
    "Zb",
    "C",
    "disruptor_count",
    "t",
    "qa",
    "qb",
)

RESULT_KEY_FIELDS = (*BASE_KEY_FIELDS, "h")


def derive_seed(base_seed: int, config: GridConfig) -> int:
    """Derive a deterministic random seed for one configuration."""
    payload = "|".join((str(base_seed), *config.key())).encode("utf-8")
    digest = hashlib.blake2s(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big")


def config_result_key(
    config: GridConfig,
    holding_time: int,
) -> tuple[str, ...]:
    return (
        *(str(getattr(config, field)) for field in BASE_KEY_FIELDS),
        str(holding_time),
    )


def row_result_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in RESULT_KEY_FIELDS)


def read_existing_results(
    output_path: Path,
) -> tuple[list[str], set[tuple[str, ...]]]:
    """Read the schema and successful result keys already in the CSV."""
    if not output_path.exists() or output_path.stat().st_size == 0:
        return [], set()

    with output_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames or []

        successful_keys = {
            row_result_key(row)
            for row in reader
            if row.get("status") == "ok"
        }

    return fields, successful_keys


def validate_experiment_definition() -> None:
    if not H_VALUES:
        raise ValueError("H_VALUES cannot be empty")

    if any(h < 0 for h in H_VALUES):
        raise ValueError("Holding times must be non-negative")

    if H_VALUES != sorted(set(H_VALUES)):
        raise ValueError(
            "H_VALUES must be sorted and contain no duplicates"
        )

    if not N_VALUES:
        raise ValueError("N_VALUES cannot be empty")

    missing_models = [
        model
        for model in MODEL_ORDER
        if model not in D_FRACTIONS_BY_MODEL
    ]
    if missing_models:
        raise ValueError(
            "Missing disruptor fractions for models: "
            f"{missing_models}"
        )

    empty_models = [
        model
        for model in MODEL_ORDER
        if not D_FRACTIONS_BY_MODEL[model]
    ]
    if empty_models:
        raise ValueError(
            "Set D_FRACTIONS_BY_MODEL before running. "
            f"Empty models: {empty_models}"
        )

    for model, fractions in D_FRACTIONS_BY_MODEL.items():
        for fraction in fractions:
            if not 0.0 <= fraction <= 1.0:
                raise ValueError(
                    f"{model}: invalid disruptor fraction {fraction}"
                )


def generate_h_sweep_configs() -> list[GridConfig]:
    """
    Generate configurations using model-specific disruptor fractions.

    GridConfig.h is set to max(H_VALUES), but the monitored holding
    times are the separate values in H_VALUES.
    """
    configs: list[GridConfig] = []

    for model in MODEL_ORDER:
        model_configs = generate_configs(
            models=[model],
            N_values=N_VALUES,
            D_fractions=D_FRACTIONS_BY_MODEL[model],
            t=T,
            h=max(H_VALUES),
            qa=QA,
            qb=QB,
        )
        configs.extend(model_configs)

    # Remove duplicates caused by different requested fractions
    # rounding to the same disruptor count.
    unique: dict[tuple[str, ...], GridConfig] = {}

    for config in configs:
        key = tuple(
            str(getattr(config, field))
            for field in BASE_KEY_FIELDS
        )
        unique[key] = config

    return list(unique.values())


def build_stability_monitors(
    reach_bound: float,
) -> dict[str, tuple[str, float, float]]:
    monitors: dict[str, tuple[str, float, float]] = {}

    for h in H_VALUES:
        monitors[f"stable_a_h{h}"] = (
            "maj_a",
            reach_bound,
            float(h),
        )
        monitors[f"stable_b_h{h}"] = (
            "maj_b",
            reach_bound,
            float(h),
        )

    return monitors


def check_monotonicity(
    successes: dict[int, dict[str, int]],
) -> None:
    """
    Since the same trajectories are used for all h values, success
    counts must be non-increasing as h increases.
    """
    for property_name in (
        "stable_a",
        "stable_b",
        "stable_consensus",
    ):
        counts = [
            successes[h][property_name]
            for h in H_VALUES
        ]

        for previous_h, next_h, previous, next_value in zip(
            H_VALUES,
            H_VALUES[1:],
            counts,
            counts[1:],
        ):
            if next_value > previous:
                raise RuntimeError(
                    f"Monotonicity violation for {property_name}: "
                    f"h={previous_h} has {previous} successes, "
                    f"but h={next_h} has {next_value}"
                )


def check_h_zero(
    successes: dict[int, dict[str, int]],
    reach_successes: dict[str, int],
) -> None:
    """
    For h=0, F<=t G<=0 label is equivalent to reaching the label
    by time t.
    """
    if 0 not in H_VALUES:
        return

    expected = {
        "stable_a": reach_successes["reach_a"],
        "stable_b": reach_successes["reach_b"],
        "stable_consensus": reach_successes["reach_consensus"],
    }

    for property_name, expected_count in expected.items():
        actual_count = successes[0][property_name]

        if actual_count != expected_count:
            raise RuntimeError(
                f"h=0 check failed for {property_name}: "
                f"stability count={actual_count}, "
                f"reachability count={expected_count}"
            )


def run_single_config(
    config: GridConfig,
    episodes: int,
    seed: int,
    confidence: float,
    show_progress: bool,
) -> list[dict[str, object]]:
    """
    Simulate each episode once and evaluate every holding time online.
    """
    params = config.to_model_params()
    rng = random.Random(seed)

    max_h = max(H_VALUES)
    simulation_horizon = params.t + max_h

    stability_monitors = build_stability_monitors(
        reach_bound=params.t,
    )

    reach_monitors = {
        "reach_a": ("maj_a", params.t),
        "reach_b": ("maj_b", params.t),
        "reach_consensus": ("consensus", params.t),
    }

    successes = {
        h: {
            "stable_a": 0,
            "stable_b": 0,
            "stable_consensus": 0,
        }
        for h in H_VALUES
    }

    reach_successes = {
        "reach_a": 0,
        "reach_b": 0,
        "reach_consensus": 0,
    }

    iterator = range(episodes)

    if show_progress:
        iterator = tqdm(
            iterator,
            unit="ep",
            leave=False,
            ncols=80,
        )

    start = time.perf_counter()

    for _ in iterator:
        results = simulate_direct_with_monitors(
            params=params,
            rng=rng,
            max_time=simulation_horizon,
            stability_monitors=stability_monitors,
            reach_monitors=reach_monitors,
        )

        for name in reach_successes:
            if results[name]:
                reach_successes[name] += 1

        for h in H_VALUES:
            stable_a = results[f"stable_a_h{h}"]
            stable_b = results[f"stable_b_h{h}"]

            if stable_a:
                successes[h]["stable_a"] += 1

            if stable_b:
                successes[h]["stable_b"] += 1

            if stable_a or stable_b:
                successes[h]["stable_consensus"] += 1

    elapsed = time.perf_counter() - start

    check_monotonicity(successes)
    check_h_zero(successes, reach_successes)

    rows: list[dict[str, object]] = []

    for h in H_VALUES:
        row: dict[str, object] = {
            "model": config.model,
            "N": config.N,
            "Za": config.Za,
            "Zb": config.Zb,
            "C": config.C,
            "disruptor_count": config.disruptor_count,
            "disruptor_fraction": config.disruptor_fraction,
            "t": config.t,
            "h": h,
            "qa": config.qa,
            "qb": config.qb,
            "simulation_horizon": simulation_horizon,
            "episodes": episodes,
            "confidence": confidence,
            "base_seed": "",
            "config_seed": seed,
            "total_time": elapsed,
            "status": "ok",
            "error": "",
        }

        for property_name in (
            "stable_a",
            "stable_b",
            "stable_consensus",
        ):
            count = successes[h][property_name]
            lower, upper = wilson_interval(
                count,
                episodes,
                confidence,
            )

            row[f"{property_name}_successes"] = count
            row[property_name] = count / episodes
            row[f"{property_name}_ci_lo"] = lower
            row[f"{property_name}_ci_hi"] = upper

        rows.append(row)

    return rows


def run_grid(
    configs: list[GridConfig],
    args,
    output_path: Path,
    overwrite: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if overwrite and output_path.exists():
        output_path.unlink()

    existing_fields, existing_keys = read_existing_results(
        output_path
    )

    if existing_fields and tuple(existing_fields) != CSV_FIELDS:
        raise ValueError(
            "Existing CSV has incompatible fields:\n"
            f"Expected: {CSV_FIELDS}\n"
            f"Found:    {tuple(existing_fields)}"
        )

    pending: list[
        tuple[GridConfig, set[int]]
    ] = []

    for config in configs:
        missing_h = {
            h
            for h in H_VALUES
            if config_result_key(config, h)
            not in existing_keys
        }

        if missing_h:
            pending.append((config, missing_h))

    print(f"Output: {output_path}")
    print(f"Total configurations: {len(configs)}")
    print(f"Configurations to run: {len(pending)}")
    print(f"N values: {N_VALUES}")
    print(f"H values: {H_VALUES}")
    print(f"Episodes per configuration: {args.episodes}")
    print(f"Confidence: {args.confidence:.3f}")
    print(f"Base seed: {args.seed}")

    if not pending:
        print("Nothing to do.")
        return

    write_header = (
        not output_path.exists()
        or output_path.stat().st_size == 0
    )

    with output_path.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
        )

        if write_header:
            writer.writeheader()

        for index, (config, missing_h) in enumerate(
            pending,
            start=1,
        ):
            print(
                f"[{index}/{len(pending)}] "
                f"{config.model}: "
                f"N={config.N}, "
                f"D={config.disruptor_count}, "
                f"rho={config.disruptor_fraction:.3f}",
                flush=True,
            )

            config_seed = derive_seed(args.seed, config)
            start = time.perf_counter()

            try:
                rows = run_single_config(
                    config=config,
                    episodes=args.episodes,
                    seed=config_seed,
                    confidence=args.confidence,
                    show_progress=not args.no_progress,
                )

                rows_by_h = {
                    int(row["h"]): row
                    for row in rows
                }

                for h in sorted(missing_h):
                    row = rows_by_h[h]
                    row["base_seed"] = args.seed
                    writer.writerow(row)

                file.flush()

                print(
                    "  ok: "
                    + ", ".join(
                        f"h={h}: "
                        f"{rows_by_h[h]['stable_consensus']:.4f}"
                        for h in H_VALUES
                    )
                    + f", {rows[0]['total_time']:.1f}s",
                    flush=True,
                )

            except Exception as error:
                elapsed = time.perf_counter() - start
                error_text = (
                    f"{type(error).__name__}: "
                    f"{' '.join(str(error).splitlines())}"
                )

                for h in sorted(missing_h):
                    error_row = {
                        "model": config.model,
                        "N": config.N,
                        "Za": config.Za,
                        "Zb": config.Zb,
                        "C": config.C,
                        "disruptor_count": config.disruptor_count,
                        "disruptor_fraction":
                            config.disruptor_fraction,
                        "t": config.t,
                        "h": h,
                        "qa": config.qa,
                        "qb": config.qb,
                        "simulation_horizon":
                            config.t + max(H_VALUES),
                        "episodes": args.episodes,
                        "confidence": args.confidence,
                        "base_seed": args.seed,
                        "config_seed": config_seed,
                        "total_time": elapsed,
                        "status": "error",
                        "error": error_text,
                    }

                    writer.writerow(error_row)

                file.flush()
                print(f"  {error_text}", flush=True)


def main() -> None:
    validate_experiment_definition()

    parser = build_sweep_parser(
        description=(
            "Statistical Model Checking sweep over holding times."
        ),
        default_episodes=4500,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the existing holding-time CSV.",
    )

    args = parser.parse_args()

    if args.episodes <= 0:
        raise ValueError("episodes must be positive")

    if not 0.0 < args.confidence < 1.0:
        raise ValueError(
            "confidence must be between 0 and 1"
        )

    configs = generate_h_sweep_configs()

    run_grid(
        configs=configs,
        args=args,
        output_path=Path("results/smc_h_sweep.csv"),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
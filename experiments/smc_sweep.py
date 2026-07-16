#!/usr/bin/env python3

import csv
import hashlib
import random
import time
from pathlib import Path

from tqdm import tqdm

from src.model import ALL_MODELS
from src.cli import build_sweep_parser, resolve_model_args
from src.gillespie_py import simulate_direct_with_monitors
from src.estimation import wilson_interval
from src.types import ModelParams

from experiments.grid_common import GridConfig, generate_configs, read_existing_csv


T = 35
H = 40

N_VALUES = [20, 50, 100, 200, 500, 1000, 2000, 4000]
D_FRACTIONS = [0, 0.1, 0.2, 0.25, 0.4, 0.5, 0.6]

CSV_FIELDS = (
    "model", "N", "Za", "Zb", "C",
    "disruptor_count", "disruptor_fraction",
    "t", "h", "qa", "qb",
    "stable_a", "stable_a_ci_lo", "stable_a_ci_hi",
    "stable_b", "stable_b_ci_lo", "stable_b_ci_hi",
    "stable_consensus", "stable_consensus_ci_lo", "stable_consensus_ci_hi",
    "reach_consensus", "reach_consensus_ci_lo", "reach_consensus_ci_hi",
    "episodes", "total_time", "status", "error",
)

STABLE_LABELS = {"stable_a": "maj_a", "stable_b": "maj_b"}


def derive_seed(base_seed: int, config: GridConfig) -> int:
    value = (
        f"{base_seed}|{config.model}|{config.N}|"
        f"{config.Za}|{config.Zb}|{config.C}"
    ).encode()

    digest = hashlib.blake2s(value, digest_size=4).digest()
    return int.from_bytes(digest, byteorder="big")

def run_single_config(
    config: GridConfig,
    episodes: int,
    seed: int,
    show_progress: bool,
) -> dict[str, object]:
    params = config.to_model_params()
    rng = random.Random(seed)

    successes = {
        "stable_a": 0,
        "stable_b": 0,
        "stable_consensus": 0,
        "reach_consensus": 0,
    }

    iterator = range(episodes)
    if show_progress:
        iterator = tqdm(iterator, unit="ep", leave=False, ncols=80)

    start = time.perf_counter()

    for _ in iterator:
        results = simulate_direct_with_monitors(
            params,
            rng,
            params.t + params.h,
            stability_monitors={
                "stable_a": ("maj_a", params.t, params.h),
                "stable_b": ("maj_b", params.t, params.h),
            },
            reach_monitors={
                "reach_consensus": ("consensus", params.t),
            },
        )

        if results["stable_a"]:
            successes["stable_a"] += 1
        if results["stable_b"]:
            successes["stable_b"] += 1
        if results["stable_a"] or results["stable_b"]:
            successes["stable_consensus"] += 1
        if results["reach_consensus"]:
            successes["reach_consensus"] += 1

    elapsed = time.perf_counter() - start

    row: dict[str, object] = {
        "model": config.model,
        "N": config.N,
        "Za": config.Za,
        "Zb": config.Zb,
        "C": config.C,
        "disruptor_count": config.disruptor_count,
        "disruptor_fraction": config.disruptor_fraction,
        "t": config.t,
        "h": config.h,
        "qa": config.qa,
        "qb": config.qb,
        "episodes": episodes,
        "total_time": elapsed,
        "status": "ok",
        "error": "",
    }

    for prop in ("stable_a", "stable_b", "stable_consensus", "reach_consensus"):
        s = successes[prop]
        lo, hi = wilson_interval(s, episodes, 0.99)
        row[prop] = s / episodes
        row[f"{prop}_ci_lo"] = lo
        row[f"{prop}_ci_hi"] = hi

    return row

def run_grid(configs: list[GridConfig], args, output_path: Path, overwrite: bool = False) -> None:
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and output_path.exists():
        output_path.unlink()

    existing_fields, existing_keys = read_existing_csv(output_path)
    if existing_fields and tuple(existing_fields) != CSV_FIELDS:
        raise ValueError(
            f"Existing CSV has incompatible fields:\n"
            f"Expected: {CSV_FIELDS}\n"
            f"Found:    {tuple(existing_fields)}"
        )
    pending = [c for c in configs if c.key() not in existing_keys]

    print(f"Output: {output_path}")
    print(f"Total configurations: {len(configs)}")
    print(f"Configurations to run: {len(pending)}")
    print(f"Episodes per config: {args.episodes}")
    print(f"Seed: {args.seed}")

    if not pending:
        print("Nothing to do.")
        return

    write_header = not output_path.exists() or output_path.stat().st_size == 0

    with output_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for index, config in enumerate(pending, start=1):
            print(
                f"[{index}/{len(pending)}] {config.model}: "
                f"N={config.N}, D={config.disruptor_count}, "
                f"rho={config.disruptor_fraction:.3f}",
                flush=True,
            )

            t0 = time.perf_counter()
            try:
                seed = derive_seed(args.seed, config)
                row = run_single_config(
                    config=config,
                    episodes=args.episodes,
                    seed=seed,
                    show_progress=not args.no_progress,
                )
                print(
                    f"  ok: stable_a={row['stable_a']:.4f}, "
                    f"stable_b={row['stable_b']:.4f}, "
                    f"stable_consensus={row['stable_consensus']:.4f}, "
                    f"reach={row['reach_consensus']:.4f}, "
                    f"{row['total_time']:.1f}s",
                    flush=True,
                )
            except Exception as error:
                total_time = time.perf_counter() - t0
                row = {
                    "model": config.model,
                    "N": config.N,
                    "Za": config.Za,
                    "Zb": config.Zb,
                    "C": config.C,
                    "disruptor_count": config.disruptor_count,
                    "disruptor_fraction": config.disruptor_fraction,
                    "t": config.t,
                    "h": config.h,
                    "qa": config.qa,
                    "qb": config.qb,
                    "episodes": args.episodes,
                    "total_time": total_time,
                    "status": "error",
                    "error": f"{type(error).__name__}: {' '.join(str(error).splitlines())}",
                }
                print(f"  {row['error']}", flush=True)

            writer.writerow(row)
            file.flush()

def main() -> None:
    parser = build_sweep_parser(
        description="Statistical Model Checking sweep using direct Gillespie simulator.",
        default_episodes=4500,
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing CSV",
    )
    args = parser.parse_args()
    args = resolve_model_args(args)

    configs = generate_configs(
        models=list(ALL_MODELS),
        N_values=N_VALUES,
        D_fractions=D_FRACTIONS,
        t=T,
        h=H,
        qa=1.05,
        qb=0.95,
    )
    run_grid(configs, args, Path("results/smc_sweep.csv"), overwrite=args.overwrite)

if __name__ == "__main__":
    main()

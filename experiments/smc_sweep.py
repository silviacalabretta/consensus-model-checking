#!/usr/bin/env python3

import csv
import hashlib
import random
import time
from pathlib import Path

from tqdm import tqdm

from src.model import ALL_MODELS
from src.cli import build_sweep_parser
from src.gillespie_py import simulate_direct_with_monitors
from src.estimation import wilson_interval

from grid_common import GridConfig, generate_configs, read_existing_csv


T = 35
H = 40

N_VALUES = [20, 50, 100, 200, 500, 1000, 2000, 4000]
D_FRACTIONS = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.28, 0.30, 0.35, 0.4, 0.45, 0.5, 0.6]

CSV_FIELDS = (
    "model", "N", "Za", "Zb", "C",
    "disruptor_count", "disruptor_fraction",
    "t", "h", "qa", "qb",
    "stable_a", "stable_a_ci_lo", "stable_a_ci_hi",
    "stable_b", "stable_b_ci_lo", "stable_b_ci_hi",
    "stable_consensus", "stable_consensus_ci_lo", "stable_consensus_ci_hi",
    "reach_a", "reach_a_ci_lo", "reach_a_ci_hi",
    "reach_b", "reach_b_ci_lo", "reach_b_ci_hi",
    "reach_consensus", "reach_consensus_ci_lo", "reach_consensus_ci_hi",
    "episodes", "confidence", "base_seed", "config_seed",
    "total_time", "status", "error",
)

STABLE_LABELS = {"stable_a": "maj_a", "stable_b": "maj_b"}
REACH_LABELS = {
    "reach_a": "maj_a",
    "reach_b": "maj_b",
    "reach_consensus": "consensus",
}
PROPERTIES = (
    "stable_a",
    "stable_b",
    "stable_consensus",
    "reach_a",
    "reach_b",
    "reach_consensus",
)



def derive_seed(base_seed: int, config: GridConfig) -> int:
    payload = "|".join((str(base_seed), *config.key())).encode()
    digest = hashlib.blake2s(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big")

def run_single_config(
    config: GridConfig,
    episodes: int,
    seed: int,
    confidence: float,
    show_progress: bool,
) -> dict[str, object]:
    params = config.to_model_params()
    rng = random.Random(seed)

    successes = {
        "stable_a": 0,
        "stable_b": 0,
        "stable_consensus": 0,
        "reach_a": 0,
        "reach_b": 0,
        "reach_consensus": 0,
    }

    iterator = range(episodes)
    if show_progress:
        iterator = tqdm(iterator, unit="ep", leave=False, ncols=80)

    start = time.perf_counter()
    stability_monitors={
        name: (label, params.t, params.h) 
        for name, label in STABLE_LABELS.items()
    }
    reach_monitors={
        name: (label, params.t) 
        for name, label in REACH_LABELS.items()
    }
    
    for _ in iterator:
        results = simulate_direct_with_monitors(
            params,
            rng,
            params.t + params.h,
            stability_monitors=stability_monitors,
            reach_monitors=reach_monitors,
        )

        if results["stable_a"]:
            successes["stable_a"] += 1
        if results["stable_b"]:
            successes["stable_b"] += 1
        if results["stable_a"] or results["stable_b"]:
            successes["stable_consensus"] += 1
        for name in REACH_LABELS:
            if results[name]:
                successes[name] += 1

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
        "confidence": confidence,
        "total_time": elapsed,
        "status": "ok",
        "error": "",
    }

    for prop in PROPERTIES:
        s = successes[prop]
        lo, hi = wilson_interval(s, episodes, confidence)
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
            seed = derive_seed(args.seed, config)
            try:
                row = run_single_config(
                    config=config,
                    episodes=args.episodes,
                    seed=seed,
                    confidence=args.confidence,
                    show_progress=not args.no_progress,
                )
                row["base_seed"] = args.seed
                row["config_seed"] = seed
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
                    "confidence": args.confidence,
                    "base_seed": args.seed,
                    "config_seed": seed,
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
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")

    if not 0.0 < args.confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

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

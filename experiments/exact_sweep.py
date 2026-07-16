#!/usr/bin/env python3

import csv
import time
from pathlib import Path

from src.checking import run_model_check
from src.model import ALL_MODELS
from src.types import ModelParams

from experiments.grid_common import GridConfig, generate_configs, read_existing_csv

T = 35
H = 40

N_VALUES = [20, 30, 50, 80, 100, 200]
D_FRACTIONS = [0, 0.1, 0.2, 0.25, 0.4, 0.5, 0.6]

BASE_CSV_FIELDS = (
    "model", "N", "Za", "Zb", "C",
    "disruptor_count", "disruptor_fraction",
    "t", "h", "qa", "qb", "nr_states", "nr_transitions",
    "build_time", "check_time", "total_time",
)


def get_csv_fields(result) -> list[str]:
    property_fields = [
        r.name for r in result.probability_results + result.expected_time_results
    ]
    return [*BASE_CSV_FIELDS, *property_fields, "status", "error"]


def values_by_name(results) -> dict[str, float]:
    return {r.name: float(r.value) for r in results}


def get_check_time(result) -> float:
    if hasattr(result, "check_time"):
        return float(result.check_time)
    all_results = result.probability_results + result.expected_time_results
    return sum(float(r.elapsed_time) for r in all_results)


def make_success_row(config: GridConfig, result, total_time: float) -> dict[str, object]:
    return {
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
        "nr_states": result.nr_states,
        "nr_transitions": result.nr_transitions,
        "build_time": result.build_time,
        "check_time": get_check_time(result),
        "total_time": total_time,
        **values_by_name(result.probability_results),
        **values_by_name(result.expected_time_results),
        "status": "ok",
        "error": "",
    }


def make_error_row(config: GridConfig, total_time: float, error: Exception) -> dict[str, object]:
    message = " ".join(str(error).splitlines())
    return {
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
        "total_time": total_time,
        "status": "error",
        "error": f"{type(error).__name__}: {message}",
    }


def run_grid(configs: list[GridConfig], output_path: Path, overwrite: bool = False) -> None:
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and output_path.exists():
        output_path.unlink()

    existing_fields, existing_keys = read_existing_csv(output_path)
    pending = [c for c in configs if c.key() not in existing_keys]

    print(f"Output: {output_path}")
    print(f"Total configurations: {len(configs)}")
    print(f"Configurations to run: {len(pending)}")

    if not pending:
        print("Nothing to do.")
        return

    write_header = not output_path.exists() or output_path.stat().st_size == 0

    with output_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=existing_fields) if existing_fields else None
        buffered_rows = []

        for index, config in enumerate(pending, start=1):
            print(
                f"[{index}/{len(pending)}] {config.model}: "
                f"N={config.N}, D={config.disruptor_count}, "
                f"rho={config.disruptor_fraction:.3f}",
                flush=True,
            )

            start = time.perf_counter()
            try:
                params = ModelParams(
                    model_name=config.model,
                    N=config.N, Za=config.Za, Zb=config.Zb,
                    C=config.C, t=config.t, h=config.h,
                    qa=config.qa, qb=config.qb,
                )
                result = run_model_check(params)
                total_time = time.perf_counter() - start

                if writer is None:
                    csv_fields = get_csv_fields(result)
                    writer = csv.DictWriter(file, fieldnames=csv_fields)
                    if write_header:
                        writer.writeheader()
                    for buffered_row in buffered_rows:
                        writer.writerow(buffered_row)
                    buffered_rows.clear()

                row = make_success_row(config, result, total_time)
                print(
                    f"  ok: {result.nr_states} states, "
                    f"{result.nr_transitions} transitions, "
                    f"{total_time:.3f}s",
                    flush=True,
                )
            except Exception as error:
                total_time = time.perf_counter() - start
                row = make_error_row(config, total_time, error)
                print(f"  {row['error']}", flush=True)

            if writer is None:
                buffered_rows.append(row)
            else:
                writer.writerow(row)
                file.flush()

        if writer is None and buffered_rows:
            raise RuntimeError("All configurations failed, no CSV property columns.")


def main() -> None:
    configs = generate_configs(
        models=list(ALL_MODELS),
        N_values=N_VALUES,
        D_fractions=D_FRACTIONS,
        t=T,
        h=H,
        qa=1.05,
        qb=0.95,
    )
    run_grid(configs, Path("results/exact_sweep.csv"), overwrite=False)


if __name__ == "__main__":
    main()

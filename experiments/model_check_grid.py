#!/usr/bin/env python3
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model_check import run_model_check

MODELS = (
    "voter_zealots",
    "voter_contrarians",
    "crossinh_zealots",
    "crossinh_contrarians",
)

BASE_CSV_FIELDS = (
    "model",
    "N",
    "Za",
    "Zb",
    "C",
    "disruptor_count",
    "disruptor_fraction",
    "t",
    "h",
    "nr_states",
    "nr_transitions",
    "build_time",
    "check_time",
    "total_time",
)

CONFIG_KEY_FIELDS = ("model", "N", "Za", "Zb", "C", "t", "h")

# fixed parameters
T = 35
H = 40

# tested parameters
N_VALUES = [20, 30, 50, 80, 100, 200, 500]
D_FRACTIONS = [0, 0.1, 0.2, 0.25, 0.4, 0.5, 0.6]   # percentage of disruptors


@dataclass(frozen=True)
class GridConfig:
    model: str
    N: int
    Za: int
    Zb: int
    C: int
    t: int
    h: int

    @property
    def disruptor_count(self) -> int:
        return self.Za + self.Zb if self.model.endswith("_zealots") else self.C

    @property
    def disruptor_fraction(self) -> float:
        return self.disruptor_count / self.N

    def key(self) -> tuple[str, ...]:
        return tuple(str(getattr(self, field)) for field in CONFIG_KEY_FIELDS)


def even_disruptor_counts(N: int, d_fractions: list[float]) -> list[int]:
    """Return the nearest even disruptor counts corresponding to the given percentages."""
    return [2 * math.floor(d * N / 2 + 0.5) for d in d_fractions]

def generate_configs(
    models: list[str],
    N_values: list[int],
    D_fractions: list[float],
    t: int,
    h: int,
) -> list[GridConfig]:
    configs: list[GridConfig] = []

    for N in sorted(set(N_values)):
        for disruptors in even_disruptor_counts(N, D_fractions):
            for model in models:
                if model.endswith("_zealots"):
                    Za = Zb = disruptors // 2
                    C = 0
                else:
                    Za = Zb = 0
                    C = disruptors

                configs.append(
                    GridConfig(
                        model=model,
                        N=N,
                        Za=Za,
                        Zb=Zb,
                        C=C,
                        t=t,
                        h=h,
                    )
                )

    return configs

def get_csv_fields(result: object) -> list[str]:
    property_fields = [property_result.name for property_result in (
            result.probability_results
            + result.expected_time_results
        )]
    return [*BASE_CSV_FIELDS, *property_fields, "status","error",]

def values_by_name(results: list[object]) -> dict[str, float]:
    """Associate results with names."""
    return {result.name: float(result.value) for result in results}

def get_check_time(result: object) -> float:
    """Support both the local version with check_time and the GitHub version."""
    if hasattr(result, "check_time"):
        return float(result.check_time)

    all_results = result.probability_results + result.expected_time_results
    return sum(float(property_result.elapsed_time) for property_result in all_results)

def make_success_row(
    config: GridConfig,
    result: object,
    total_time: float,
) -> dict[str, object]:
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


def make_error_row(
    config: GridConfig,
    total_time: float,
    error: Exception,
) -> dict[str, object]:
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
        "total_time": total_time,
        "status": "error",
        "error": f"{type(error).__name__}: {message}",
    }


# def read_existing_keys(output_path: Path) -> set[tuple[str, ...]]:
#     """Read already written configurations, allowing an interrupted run to resume."""
#     if not output_path.exists() or output_path.stat().st_size == 0:
#         return set()

#     with output_path.open("r", encoding="utf-8", newline="") as file:
#         reader = csv.DictReader(file)
#         return {
#             tuple(row[field] for field in CONFIG_KEY_FIELDS)
#             for row in reader
#         }

# def read_csv_fields(output_path: Path) -> list[str] | None:
#     if not output_path.exists() or output_path.stat().st_size == 0:
#         return None

#     with output_path.open("r", encoding="utf-8", newline="") as file:
#         return csv.DictReader(file).fieldnames

def read_existing_csv(output_path: Path) -> tuple[list[str] | None, set[tuple[str, ...]]]:
    """Return the CSV header and the configurations already recorded."""
    if not output_path.exists() or output_path.stat().st_size == 0:
        return None, set()

    with output_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        fields = reader.fieldnames
        keys = {
            tuple(row[field] for field in CONFIG_KEY_FIELDS)
            for row in reader
        }

    return fields, keys

def run_grid(
    configs: list[GridConfig],
    output_path: Path,
    overwrite: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if overwrite and output_path.exists():
        output_path.unlink()

    # existing_keys = read_existing_keys(output_path)
    existing_fields, existing_keys = read_existing_csv(output_path)
    pending = [config for config in configs if config.key() not in existing_keys]

    print(f"Output: {output_path}")
    print(f"Total configurations: {len(configs)}")
    print(f"Configurations to run: {len(pending)}")

    if not pending:
        print("Nothing to do.")
        return

    write_header = not output_path.exists() or output_path.stat().st_size == 0

    with output_path.open("a", encoding="utf-8", newline="") as file:
        writer = (
            csv.DictWriter(file, fieldnames=existing_fields) if existing_fields else None
        )
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
                result = run_model_check(
                    model_name=config.model,
                    N=config.N,
                    Za=config.Za,
                    Zb=config.Zb,
                    C=config.C,
                    t=config.t,
                    h=config.h,
                )
                total_time = time.perf_counter() - start

                if writer is None:
                    # define writer by getting property names from results
                    csv_fields = get_csv_fields(result)
                    writer = csv.DictWriter(file, fieldnames=csv_fields)
                    if write_header:
                        writer.writeheader()
                    
                    # add error rows occurred before first success
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

            # Write and flush immediately, so previous results survive a crash.
            if writer is None:
                buffered_rows.append(row)
            else:
                writer.writerow(row)
                file.flush()

        if writer is None and buffered_rows:
            raise RuntimeError("All configurations failed, no CSV property columns.")


def main() -> None:
    overwrite=False
    configs = generate_configs(
        models=MODELS,
        N_values=N_VALUES,
        D_fractions=D_FRACTIONS,
        t=T,
        h=H,
    )
    run_grid(configs, ROOT_DIR / "results" / "model_check_grid.csv",overwrite=overwrite)


if __name__ == "__main__":
    main()
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.types import EpisodeResult

MODEL_LABELS = {
    "voter_zealots": "Voter + zealots",
    "voter_contrarians": "Voter + contrarians",
    "crossinh_zealots": "Cross-inhibition + zealots",
    "crossinh_contrarians": "Cross-inhibition + contrarians",
}

SCALABILITY_METRICS = {
    "nr_states": "Reachable states",
    "nr_transitions": "Transitions",
    "build_time": "Build time [s]",
    "check_time": "Model-checking time [s]",
    "total_time": "Total time [s]",
}

NON_PROPERTY_FIELDS = {
    "model", "N", "Za", "Zb", "C",
    "disruptor_count", "disruptor_fraction",
    "target_disruptor_fraction",
    "t", "h", "qa", "qb",
    "nr_states", "nr_transitions",
    "build_time", "check_time", "total_time",
    "status", "error",
}


def load_results(csv_path: Path) -> pd.DataFrame:
    data = pd.read_csv(csv_path)

    if "status" in data.columns:
        data = data[data["status"] == "ok"].copy()

    if data.empty:
        raise ValueError("The CSV contains no successful configurations.")

    numeric_columns = [
        col for col in data.columns
        if col not in NON_PROPERTY_FIELDS
        and pd.api.types.is_numeric_dtype(data[col])
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(
        subset=["model", "N", "disruptor_fraction"]
    )

    return data.sort_values(
        ["model", "N", "disruptor_fraction"]
    ).reset_index(drop=True)


def property_fields(data: pd.DataFrame) -> list[str]:
    return [
        column
        for column in data.columns
        if column not in NON_PROPERTY_FIELDS
        and pd.api.types.is_numeric_dtype(data[column])
    ]


def fraction_field(data: pd.DataFrame) -> str:
    if "target_disruptor_fraction" in data.columns:
        return "target_disruptor_fraction"
    return "disruptor_fraction"


def format_grid_axis(ax: plt.Axes, data: pd.DataFrame, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("Actual disruptor fraction $D/N$")
    ax.set_ylabel("Population size $N$")
    ax.set_yticks(sorted(data["N"].unique()))
    ax.grid(True, alpha=0.25)


def safe_filename(text: str) -> str:
    return text.lower().replace(" ", "_").replace("/", "_")


def _sample_trajectory_on_grid(
    result: EpisodeResult,
    variable: str,
    time_grid: np.ndarray,
) -> np.ndarray:
    transition_times = np.asarray(
        [step.time for step in result.steps], dtype=float,
    )
    values = np.asarray(
        [step.variables[variable] for step in result.steps], dtype=float,
    )
    indices = np.searchsorted(
        transition_times, time_grid, side="right",
    ) - 1
    indices = np.clip(indices, 0, len(transition_times) - 1)
    return values[indices]


def plot_trajectories(
    data: EpisodeResult | Sequence[EpisodeResult],
    variables: Optional[Sequence[str]] = None,
    *,
    grid_points: int = 500,
    shade_labels: bool = False,
    show: bool = False,
    save_path: Optional[str | Path] = None,
):
    if isinstance(data, EpisodeResult):
        results = [data]
    else:
        results = list(data)

    if not results:
        raise ValueError("At least one trajectory is required")

    if any(not result.steps for result in results):
        raise ValueError("All trajectories must contain at least one step")

    avail_vars = list(results[0].steps[0].variables)
    selected_variables = avail_vars if variables is None else list(variables)

    if not selected_variables:
        raise ValueError("At least one variable must be selected")

    for episode_index, result in enumerate(results):
        episode_variables = result.steps[0].variables
        missing = [
            v for v in selected_variables if v not in episode_variables
        ]
        if missing:
            raise ValueError(
                f"Trajectory {episode_index} does not contain variables: {missing}"
            )

    if len(results) == 1:
        result = results[0]
        fig, ax = plt.subplots(figsize=(10, 5))
        variable_colors: dict[str, str] = {}

        for variable in selected_variables:
            times = [step.time for step in result.steps]
            values = [step.variables[variable] for step in result.steps]

            if result.final_time > times[-1]:
                times = [*times, result.final_time]
                values = [*values, values[-1]]

            line, = ax.step(times, values, where="post", label=variable, zorder=2)
            variable_colors[variable] = line.get_color()

        if shade_labels:
            maj_a_color = variable_colors.get("a", "tab:blue")
            maj_b_color = variable_colors.get("b", "tab:orange")
            used_legend_labels: set[str] = set()

            for i, step in enumerate(result.steps):
                interval_start = step.time
                interval_end = (
                    result.steps[i + 1].time if i + 1 < len(result.steps)
                    else result.final_time
                )
                if interval_end <= interval_start:
                    continue

                color = None
                legend_label = None

                if "maj_a" in step.labels:
                    color = maj_a_color
                    legend_label = "A majority"
                elif "maj_b" in step.labels:
                    color = maj_b_color
                    legend_label = "B majority"
                elif "consensus" in step.labels:
                    color = "green"
                    legend_label = "Consensus"

                if color is None:
                    continue

                displayed_label = (
                    legend_label
                    if legend_label not in used_legend_labels
                    else "_nolegend_"
                )

                ax.axvspan(
                    interval_start, interval_end,
                    color=color, alpha=0.15,
                    label=displayed_label, zorder=0,
                )
                used_legend_labels.add(legend_label)

        ax.set_title("Simulated CTMC trajectory")
        ax.set_xlabel("Time")
        ax.set_ylabel("Population")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        if show:
            plt.show()
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig, (ax,)

    if grid_points < 2:
        raise ValueError("grid_points must be at least 2")

    final_time = min(result.final_time for result in results)
    if final_time <= 0.0:
        raise ValueError("Trajectories must have a positive final time")

    time_grid = np.linspace(0.0, final_time, grid_points)
    fig, (mean_ax, variance_ax) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for variable in selected_variables:
        sampled_values = np.vstack([
            _sample_trajectory_on_grid(result, variable, time_grid)
            for result in results
        ])
        mean = sampled_values.mean(axis=0)
        variance = sampled_values.var(axis=0, ddof=1)
        standard_deviation = np.sqrt(variance)

        mean_line, = mean_ax.plot(time_grid, mean, label=variable)
        mean_ax.fill_between(
            time_grid,
            mean - standard_deviation,
            mean + standard_deviation,
            alpha=0.2, color=mean_line.get_color(),
        )
        variance_ax.plot(
            time_grid, variance,
            label=variable, color=mean_line.get_color(),
        )

    mean_ax.set_title(f"Mean trajectories over {len(results)} simulations")
    mean_ax.set_ylabel("Mean population")
    mean_ax.grid(True, alpha=0.3)
    mean_ax.legend()

    variance_ax.set_title("Variance across trajectories")
    variance_ax.set_xlabel("Time")
    variance_ax.set_ylabel("Variance")
    variance_ax.grid(True, alpha=0.3)
    variance_ax.legend()

    fig.tight_layout()

    if show:
        plt.show()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, (mean_ax, variance_ax)

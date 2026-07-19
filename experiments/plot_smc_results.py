#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter


MODEL_ORDER = (
    "voter_zealots",
    "voter_contrarians",
    "crossinh_zealots",
    "crossinh_contrarians",
)

MODEL_LABELS = {
    "voter_zealots": "Voter + zealots",
    "voter_contrarians": "Voter + contrarians",
    "crossinh_zealots": "Cross-inhibition + zealots",
    "crossinh_contrarians": "Cross-inhibition + contrarians",
}

REQUIRED_COLUMNS = {
    "model",
    "N",
    "disruptor_count",
    "disruptor_fraction",
    "stable_a",
    "stable_a_ci_lo",
    "stable_a_ci_hi",
    "stable_b",
    "stable_b_ci_lo",
    "stable_b_ci_hi",
    "stable_consensus",
    "stable_consensus_ci_lo",
    "stable_consensus_ci_hi",
    "reach_consensus",
    "reach_consensus_ci_lo",
    "reach_consensus_ci_hi",
    "episodes",
    "total_time",
    "status",
}

plt.rcParams.update({
    "font.size": 14,          # The base font size for everything
    "axes.titlesize": 16,     # The titles of individual subplots
    "axes.labelsize": 14,     # X and Y axis labels
    "xtick.labelsize": 12,    # X-axis tick labels
    "ytick.labelsize": 12,    # Y-axis tick labels
    "legend.fontsize": 12,    # Legend text
    "figure.titlesize": 18    # The main figure suptitle
})

def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    data = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(
            f"Missing required CSV columns: {sorted(missing)}"
        )

    data = data[data["status"] == "ok"].copy()

    if data.empty:
        raise ValueError("The CSV contains no successful configurations")

    numeric_columns = REQUIRED_COLUMNS - {"model", "status"}
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="raise")

    unknown_models = set(data["model"]) - set(MODEL_ORDER)
    if unknown_models:
        raise ValueError(
            f"Unknown models in CSV: {sorted(unknown_models)}"
        )

    duplicate_key = ["model", "N", "disruptor_count"]
    duplicates = data.duplicated(duplicate_key, keep=False)

    if duplicates.any():
        count = int(duplicates.sum())
        print(
            f"Warning: found {count} rows belonging to duplicated "
            "configurations; keeping the last row for each configuration."
        )
        data = data.drop_duplicates(
            duplicate_key,
            keep="last",
        )

    return data.sort_values(
        ["model", "N", "disruptor_fraction"]
    ).reset_index(drop=True)


def save_figure(
    figure: plt.Figure,
    output_dir: Path,
    filename: str,
    dpi: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / f"{filename}.png"
    # pdf_path = output_dir / f"{filename}.pdf"

    figure.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
    )
    # figure.savefig(
    #     pdf_path,
    #     bbox_inches="tight",
    # )

    plt.close(figure)

    print(f"Saved: {png_path}")
    # print(f"Saved: {pdf_path}")


def model_subset(
    data: pd.DataFrame,
    model: str,
    N: int | None = None,
) -> pd.DataFrame:
    subset = data[data["model"] == model]

    if N is not None:
        subset = subset[subset["N"] == N]

    return subset.sort_values("disruptor_fraction")


def plot_paper_style_stability(
    data: pd.DataFrame,
    output_dir: Path,
    N: int,
    dpi: int,
) -> None:
    """Paper-style Figure 3: stable A/B consensus at fixed N."""
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11, 8),
        sharex=True,
        sharey=True,
    )

    a_handle = None
    b_handle = None

    for axis, model in zip(axes.flat, MODEL_ORDER):
        subset = model_subset(data, model, N)

        if subset.empty:
            axis.set_visible(False)
            continue

        x = subset["disruptor_fraction"].to_numpy()

        a_handle = axis.plot(
            x,
            subset["stable_a"],
            marker="o",
            linewidth=2,
            label="Stable A consensus",
            color="tab:red",
        )[0]

        axis.fill_between(
            x,
            subset["stable_a_ci_lo"],
            subset["stable_a_ci_hi"],
            alpha=0.18,
            color="tab:red",
        )

        b_handle = axis.plot(
            x,
            subset["stable_b"],
            marker="s",
            linestyle="--",
            linewidth=2,
            label="Stable B consensus",
            color="tab:blue",
        )[0]

        axis.fill_between(
            x,
            subset["stable_b_ci_lo"],
            subset["stable_b_ci_hi"],
            alpha=0.18,
            color="tab:blue",
        )

        axis.set_title(MODEL_LABELS[model])
        axis.set_xlim(0.0, 0.6)
        axis.set_ylim(0.0, 1.02)
        axis.grid(alpha=0.3)

    figure.suptitle(
        f"Stable consensus probabilities, N={N}",
    )
    figure.supxlabel("Disruptor fraction", y=0.02)
    figure.supylabel("Estimated probability", x=0.02)

    if a_handle is not None and b_handle is not None:
        figure.legend(
            handles=[a_handle, b_handle],
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(0.5, 0.947),
        )

    # figure.tight_layout(rect=(0.03, 0.03, 1.0, 0.95))
    figure.subplots_adjust(
            left=0.08,
            right=0.98,
            bottom=0.1,
            top=0.86,
            wspace=0.08,
            hspace=0.22,
        )
    
    save_figure(
        figure,
        output_dir,
        f"smc_stable_consensus_N{N}",
        dpi,
    )


def plot_finite_size_scaling(
    data: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    """Paper-style Figure 4: stable A probability for different N."""
    N_values = sorted(data["N"].unique())

    colours = plt.cm.viridis(
        np.linspace(0.05, 0.95, len(N_values))
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11.5, 7.8),
        sharex=True,
        sharey=True,
    )

    handles = []

    for axis, model in zip(axes.flat, MODEL_ORDER):
        for index, N in enumerate(N_values):
            subset = model_subset(data, model, int(N))

            if subset.empty:
                continue

            line = axis.plot(
                subset["disruptor_fraction"],
                subset["stable_a"],
                marker="o",
                markersize=4,
                linewidth=1.7,
                color=colours[index],
                label=f"N={int(N)}",
            )[0]

            if len(handles) < len(N_values):
                handles.append(line)

        axis.set_title(MODEL_LABELS[model])
        axis.set_xlim(0.0, 0.6)
        axis.set_ylim(0.0, 1.02)
        axis.grid(alpha=0.3)

    figure.suptitle(
        "Finite-size scaling of stable A consensus",
        y=0.99,
    )
    figure.supxlabel("Disruptor fraction", y=0.02)
    figure.supylabel("Estimated probability", x=0.025)

    figure.legend(
        handles=handles,
        labels=[f"N={int(N)}" for N in N_values],
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.87),
    )

    # figure.tight_layout(rect=(0.03, 0.08, 1.0, 0.94))
    figure.subplots_adjust(
        left=0.1,
        right=0.98,
        bottom=0.10,
        top=0.84,
        wspace=0.08,
        hspace=0.20,
    )
    
    save_figure(
        figure,
        output_dir,
        "smc_finite_size_stable_a",
        dpi,
    )


def plot_reach_vs_stability(
    data: pd.DataFrame,
    output_dir: Path,
    N: int,
    dpi: int,
) -> None:
    """Compare reaching consensus with maintaining it."""
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11, 8),
        sharex=True,
        sharey=True,
    )

    reach_handle = None
    stable_handle = None

    for axis, model in zip(axes.flat, MODEL_ORDER):
        subset = model_subset(data, model, N)

        if subset.empty:
            axis.set_visible(False)
            continue

        x = subset["disruptor_fraction"].to_numpy()

        reach_handle = axis.plot(
            x,
            subset["reach_consensus"],
            marker="o",
            linewidth=2,
            label="Reach consensus",
            color="tab:green",
        )[0]

        axis.fill_between(
            x,
            subset["reach_consensus_ci_lo"],
            subset["reach_consensus_ci_hi"],
            alpha=0.15,
            color="tab:green",
        )

        stable_handle = axis.plot(
            x,
            subset["stable_consensus"],
            marker="s",
            linestyle="--",
            linewidth=2,
            label="Stable consensus",
            color="tab:purple",
        )[0]

        axis.fill_between(
            x,
            subset["stable_consensus_ci_lo"],
            subset["stable_consensus_ci_hi"],
            alpha=0.15,
            color="tab:purple",
        )

        axis.set_title(MODEL_LABELS[model])
        axis.set_xlim(0.0, 0.6)
        axis.set_ylim(0.0, 1.02)
        axis.grid(alpha=0.3)

    figure.suptitle(
        f"Reaching versus maintaining consensus, N={N}",
    )
    figure.supxlabel("Disruptor fraction",y=0.025)
    figure.supylabel("Estimated probability",x=0.02)

    if reach_handle is not None and stable_handle is not None:
        figure.legend(
            handles=[reach_handle, stable_handle],
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(0.5, 0.945),
        )

    # figure.tight_layout(rect=(0.03, 0.03, 1.0, 0.95))
    figure.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.1,
        top=0.86,
        wspace=0.08,
        hspace=0.22,
    )

    save_figure(
        figure,
        output_dir,
        f"smc_reach_vs_stability_N{N}",
        dpi,
    )

def plot_phase_diagrams(
    data: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    """
    Plot interpolation-free phase diagrams at the actual sampled
    disruptor fractions.
    """
    N_values = sorted(data["N"].unique())

    figure = plt.figure(figsize=(11.5, 8.5))

    grid = figure.add_gridspec(
        nrows=2,
        ncols=3,
        width_ratios=[1.0, 1.0, 0.045],
        left=0.08,
        right=0.92,
        bottom=0.10,
        top=0.90,
        wspace=0.20,
        hspace=0.25,
    )

    axes = np.array([
        [
            figure.add_subplot(grid[0, 0]),
            figure.add_subplot(grid[0, 1]),
        ],
        [
            figure.add_subplot(grid[1, 0]),
            figure.add_subplot(grid[1, 1]),
        ],
    ])

    colorbar_axis = figure.add_subplot(grid[:, 2])

    scatter = None
    
    for axis, model in zip(axes.flat, MODEL_ORDER):
        subset = model_subset(data, model)

        scatter = axis.scatter(
            subset["disruptor_fraction"],
            subset["N"],
            c=subset["stable_a"],
            marker="s",
            s=135,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            edgecolors="white",
            linewidths=0.25,
        )

        axis.set_title(
            MODEL_LABELS[model],
            pad=8,
        )

        axis.set_xlim(-0.02, 0.62)
        axis.set_yscale("log")
        axis.set_yticks(N_values)
        axis.set_xticks(np.arange(0.0, 0.61, 0.1))
        axis.set_xticks(np.arange(0.05, 0.61, 0.1), minor=True)
        axis.yaxis.set_major_formatter(ScalarFormatter())
        axis.grid(alpha=0.20)
        axis.set_axisbelow(True)
        axis.grid(axis="x",which="minor",alpha=0.20)
        axis.tick_params(axis="x",which="minor",labelbottom=False)

    # Show y tick labels only on the left column.
    axes[0, 1].tick_params(labelleft=False)
    axes[1, 1].tick_params(labelleft=False)
    
    figure.suptitle(
        "Stable A consensus phase diagrams",
        y=0.975,
    )

    figure.supxlabel(
        "Disruptor fraction",
        y=0.035,
    )

    figure.supylabel(
        "Population size N",
        x=0.015,
    )

    if scatter is not None:
        colorbar = figure.colorbar(
            scatter,
            cax=colorbar_axis,
        )
        colorbar.set_label(
            "Estimated probability of stable A consensus",
            labelpad=10,
        )

    save_figure(
        figure,
        output_dir,
        "smc_phase_diagrams_stable_a",
        dpi,
    )


def plot_runtime_scaling(
    data: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    """Plot median simulation cost per episode versus N."""
    runtime = data.copy()
    runtime["time_per_episode"] = (
        runtime["total_time"] / runtime["episodes"]
    )

    grouped = (
        runtime.groupby(["model", "N"])["time_per_episode"]
        .agg(["median", "min", "max"])
        .reset_index()
    )

    figure, axis = plt.subplots(figsize=(8, 5.5))

    for model in MODEL_ORDER:
        subset = grouped[grouped["model"] == model].sort_values("N")

        if subset.empty:
            continue

        x = subset["N"].to_numpy()
        median = subset["median"].to_numpy()
        minimum = subset["min"].to_numpy()
        maximum = subset["max"].to_numpy()

        axis.plot(
            x,
            median,
            marker="o",
            linewidth=2,
            label=MODEL_LABELS[model],
        )

        axis.fill_between(
            x,
            minimum,
            maximum,
            alpha=0.12,
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Population size N")
    axis.set_ylabel("Simulation time per episode [s]")
    axis.set_title("Scalability of the direct Python simulator")
    axis.grid(alpha=0.3, which="both")
    axis.legend()

    figure.tight_layout()

    save_figure(
        figure,
        output_dir,
        "smc_runtime_scaling",
        dpi,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate all plots from the SMC sweep results."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/smc_sweep.csv"),
        help="Input CSV generated by smc_sweep.py.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/plots/smc"),
        help="Directory in which plots are saved.",
    )

    parser.add_argument(
        "--reference-N",
        type=int,
        default=100,
        help="Population size used for fixed-N plots.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution of PNG files.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    data = load_results(args.input)

    available_N = set(data["N"].astype(int))
    if args.reference_N not in available_N:
        raise ValueError(
            f"N={args.reference_N} is not present in the CSV. "
            f"Available values: {sorted(available_N)}"
        )

    print(f"Loaded {len(data)} successful configurations")
    print(f"Input: {args.input}")
    print(f"Output directory: {args.output_dir}")

    plot_paper_style_stability(
        data,
        args.output_dir,
        args.reference_N,
        args.dpi,
    )

    plot_finite_size_scaling(
        data,
        args.output_dir,
        args.dpi,
    )

    plot_reach_vs_stability(
        data,
        args.output_dir,
        args.reference_N,
        args.dpi,
    )

    plot_phase_diagrams(
        data,
        args.output_dir,
        args.dpi,
    )

    plot_runtime_scaling(
        data,
        args.output_dir,
        args.dpi,
    )


if __name__ == "__main__":
    main()
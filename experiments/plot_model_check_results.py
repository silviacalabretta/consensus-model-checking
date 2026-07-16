#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize

from src.cli import build_plot_parser
from src.model import ALL_MODELS
from src.plotting import (
    MODEL_LABELS,
    SCALABILITY_METRICS,
    format_grid_axis,
    load_results,
)

MODEL_ORDER = ALL_MODELS

REQUIRED_PROPERTIES = (
    "reach_a",
    "reach_b",
    "reach_consensus",
    "reach_robust95_a",
    "reach_robust95_b",
    "reach_robust99_a",
    "reach_robust99_b",
    "exp_consensus_time",
)


def model_data(data, model: str):
    return data[data["model"] == model]


def plot_consensus_reachability(data, output_dir: Path, show: bool) -> None:
    fig, axes = plt.subplots(
        2, 2, figsize=(12, 9),
        sharex=True, sharey=True, layout="constrained",
    )
    axes = axes.ravel()
    scatter = None

    for ax, model in zip(axes, MODEL_ORDER, strict=True):
        subset = model_data(data, model)
        scatter = ax.scatter(
            subset["disruptor_fraction"], subset["N"],
            c=subset["reach_consensus"], vmin=0, vmax=1, s=105,
        )
        format_grid_axis(ax, data, MODEL_LABELS[model])

    fig.suptitle("Probability of reaching consensus within the time bound")
    fig.colorbar(scatter, ax=axes.tolist(), label="Probability")
    fig.savefig(output_dir / "01_reach_consensus.png", dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def plot_probability_comparison(
    data, properties: tuple[str, ...], property_labels: tuple[str, ...],
    title: str, filename: str, output_dir: Path, show: bool,
) -> None:
    fig, axes = plt.subplots(
        len(MODEL_ORDER), len(properties),
        figsize=(5 * len(properties), 3.2 * len(MODEL_ORDER)),
        sharex=True, sharey=True, layout="constrained", squeeze=False,
    )
    scatter = None

    for row, model in enumerate(MODEL_ORDER):
        subset = model_data(data, model)
        for column, (property_name, property_label) in enumerate(
            zip(properties, property_labels, strict=True)
        ):
            ax = axes[row, column]
            scatter = ax.scatter(
                subset["disruptor_fraction"], subset["N"],
                c=subset[property_name], vmin=0, vmax=1, s=95,
            )
            if row == 0:
                ax.set_title(property_label)
            if column == 0:
                ax.set_ylabel(f"{MODEL_LABELS[model]}\nPopulation size $N$")
            if row == len(MODEL_ORDER) - 1:
                ax.set_xlabel("Actual disruptor fraction $D/N$")
            ax.set_yticks(sorted(data["N"].unique()))
            ax.grid(True, alpha=0.25)

    fig.suptitle(title)
    fig.colorbar(scatter, ax=axes.ravel().tolist(), label="Probability")
    fig.savefig(output_dir / filename, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def plot_expected_consensus_time(data, output_dir: Path, show: bool) -> None:
    values = data["exp_consensus_time"].to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("No finite expected-consensus-time values to plot.")

    positive_values = finite_values[finite_values > 0]
    use_log_scale = (
        positive_values.size == finite_values.size
        and positive_values.max() / positive_values.min() >= 20
    )

    if use_log_scale:
        norm = LogNorm(vmin=float(positive_values.min()), vmax=float(positive_values.max()))
    else:
        norm = Normalize(vmin=float(finite_values.min()), vmax=float(finite_values.max()))

    fig, axes = plt.subplots(
        2, 2, figsize=(12, 9),
        sharex=True, sharey=True, layout="constrained",
    )
    axes = axes.ravel()
    scatter = None
    has_nonfinite = False

    for ax, model in zip(axes, MODEL_ORDER, strict=True):
        subset = model_data(data, model)
        finite = np.isfinite(subset["exp_consensus_time"])
        finite_subset = subset[finite]
        scatter = ax.scatter(
            finite_subset["disruptor_fraction"], finite_subset["N"],
            c=finite_subset["exp_consensus_time"], norm=norm, s=105,
        )
        nonfinite_subset = subset[~finite]
        if not nonfinite_subset.empty:
            has_nonfinite = True
            ax.scatter(
                nonfinite_subset["disruptor_fraction"], nonfinite_subset["N"],
                marker="x", s=85,
            )
        format_grid_axis(ax, data, MODEL_LABELS[model])

    scale_text = "logarithmic colour scale" if use_log_scale else "linear colour scale"
    fig.suptitle(f"Expected time to reach consensus ({scale_text})")
    fig.colorbar(scatter, ax=axes.tolist(), label="Expected consensus time")

    if has_nonfinite:
        fig.text(
            0.5, 0.01,
            "\u00d7 denotes an infinite or undefined expected time.",
            ha="center",
        )

    fig.savefig(output_dir / "04_expected_consensus_time.png", dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def plot_scalability(data, output_dir: Path, show: bool) -> None:
    fig, axes = plt.subplots(
        2, 2, figsize=(13, 9),
        sharex=True, layout="constrained",
    )
    axes = axes.ravel()

    for ax, (metric, label) in zip(axes, SCALABILITY_METRICS.items(), strict=True):
        for model in MODEL_ORDER:
            summary = (
                model_data(data, model)
                .groupby("N")[metric]
                .agg(["min", "median", "max"])
                .reset_index()
            )
            lower = summary["median"] - summary["min"]
            upper = summary["max"] - summary["median"]
            ax.errorbar(
                summary["N"], summary["median"],
                yerr=np.vstack([lower, upper]),
                marker="o", capsize=3, label=MODEL_LABELS[model],
            )
        ax.set_title(label)
        ax.set_xlabel("Population size $N$")
        ax.set_ylabel(label)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        "Scalability of exact model checking\n"
        "Median and min-max range over disruptor levels"
    )
    fig.savefig(output_dir / "05_scalability.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def create_plots(csv_path: Path, output_dir: Path, show: bool = False) -> None:
    data = load_results(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_consensus_reachability(data=data, output_dir=output_dir, show=show)

    plot_probability_comparison(
        data=data,
        properties=("reach_a", "reach_b"),
        property_labels=("Reach $A$ majority", "Reach $B$ majority"),
        title="Probability of reaching each majority within the time bound",
        filename="02_reach_a_vs_b.png",
        output_dir=output_dir,
        show=show,
    )

    plot_probability_comparison(
        data=data,
        properties=("reach_robust95_a", "reach_robust95_b", "reach_robust99_a", "reach_robust99_b"),
        property_labels=(
            "$A$: 95% robust", "$B$: 95% robust",
            "$A$: 99% robust", "$B$: 99% robust",
        ),
        title="Probability of reaching a probabilistically robust majority",
        filename="03_robustness_overview.png",
        output_dir=output_dir,
        show=show,
    )

    plot_expected_consensus_time(data=data, output_dir=output_dir, show=show)
    plot_scalability(data=data, output_dir=output_dir, show=show)

    print(f"Plots saved in: {output_dir}")


def main() -> None:
    parser = build_plot_parser("Plot exact model-checking grid results.")
    args = parser.parse_args()
    create_plots(csv_path=args.input, output_dir=args.output_dir, show=args.show)


if __name__ == "__main__":
    main()

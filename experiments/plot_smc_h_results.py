#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 14,          # The base font size for everything
    "axes.titlesize": 16,     # The titles of individual subplots
    "axes.labelsize": 14,     # X and Y axis labels
    "xtick.labelsize": 12,    # X-axis tick labels
    "ytick.labelsize": 12,    # Y-axis tick labels
    "legend.fontsize": 12,    # Legend text
    "figure.titlesize": 18    # The main figure suptitle
})

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
    "h",
    "stable_a",
    "stable_a_ci_lo",
    "stable_a_ci_hi",
    "episodes",
    "confidence",
    "status",
}


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
        raise ValueError(
            "The CSV contains no successful configurations"
        )

    numeric_columns = REQUIRED_COLUMNS - {"model", "status"}

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="raise",
        )

    unknown_models = set(data["model"]) - set(MODEL_ORDER)
    if unknown_models:
        raise ValueError(
            f"Unknown models in CSV: {sorted(unknown_models)}"
        )

    key_columns = [
        "model",
        "N",
        "disruptor_count",
        "h",
    ]

    duplicates = data.duplicated(
        key_columns,
        keep=False,
    )

    if duplicates.any():
        print(
            "Warning: duplicated rows found; keeping the last "
            "successful result for each configuration."
        )

        data = data.drop_duplicates(
            key_columns,
            keep="last",
        )

    data = data.sort_values(
        [
            "model",
            "N",
            "disruptor_fraction",
            "h",
        ]
    ).reset_index(drop=True)

    check_monotonicity(data)

    return data


def check_monotonicity(data: pd.DataFrame) -> None:
    """
    The estimated probability must be non-increasing in h because
    the same trajectories are used for all holding times.
    """
    group_columns = [
        "model",
        "N",
        "disruptor_count",
    ]

    violations = []

    for key, group in data.groupby(group_columns):
        group = group.sort_values("h")
        values = group["stable_a"].to_numpy()

        if np.any(np.diff(values) > 1e-12):
            violations.append(key)

    if violations:
        raise ValueError(
            "Stable-consensus probability increases with h for "
            f"the following configurations: {violations}"
        )


def format_fraction(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def save_figure(
    figure: plt.Figure,
    output_dir: Path,
    filename: str,
    dpi: int,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

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


def plot_stability_over_h(
    data: pd.DataFrame,
    output_dir: Path,
    N: int,
    reference_h: float,
    dpi: int,
) -> None:
    """
    Main plot: stable A-consensus probability as a function of h.
    """
    population_data = data[data["N"] == N]

    if population_data.empty:
        raise ValueError(f"No results found for N={N}")

    h_values = sorted(
        population_data["h"].unique()
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11.5, 7.8),
        sharex=True,
        sharey=True,
    )

    for axis, model in zip(
        axes.flat,
        MODEL_ORDER,
    ):
        model_data = population_data[
            population_data["model"] == model
        ]

        fractions = sorted(
            model_data["disruptor_fraction"].unique()
        )

        colours = plt.cm.viridis(
            np.linspace(
                0.08,
                0.92,
                len(fractions),
            )
        )

        for colour, fraction in zip(
            colours,
            fractions,
        ):
            subset = model_data[
                np.isclose(
                    model_data["disruptor_fraction"],
                    fraction,
                )
            ].sort_values("h")

            x = subset["h"].to_numpy()
            estimate = subset[
                "stable_a"
            ].to_numpy()
            lower = subset[
                "stable_a_ci_lo"
            ].to_numpy()
            upper = subset[
                "stable_a_ci_hi"
            ].to_numpy()

            axis.plot(
                x,
                estimate,
                marker="o",
                linewidth=1.8,
                markersize=4,
                color=colour,
                label=(
                    r"$\rho="
                    + format_fraction(fraction)
                    + "$"
                ),
            )

            axis.fill_between(
                x,
                lower,
                upper,
                color=colour,
                alpha=0.12,
            )

        if min(h_values) <= reference_h <= max(h_values):
            axis.axvline(
                reference_h,
                linestyle=":",
                linewidth=1.2,
                color="0.35",
            )

        axis.set_title(
            MODEL_LABELS[model],
        )

        axis.set_xticks(h_values)
        axis.set_xlim(
            min(h_values),
            max(h_values),
        )
        axis.set_ylim(0.0, 1.02)
        axis.grid(alpha=0.25)

        axis.legend(
            fontsize=8,
            ncol=2,
            loc="best",
        )

    figure.suptitle(
        f"Stable A-consensus versus holding time, N={N}",
        y=0.98,
    )

    figure.supxlabel(
        "Required holding time $h$",
        y=0.025,
    )

    figure.supylabel(
        "Estimated probability",
        x=0.025,
    )

    figure.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.10,
        top=0.91,
        wspace=0.08,
        hspace=0.22,
    )

    save_figure(
        figure,
        output_dir,
        f"smc_h_stable_a_N{N}",
        dpi,
    )


def estimate_h_threshold(
    group: pd.DataFrame,
    probability_threshold: float,
) -> tuple[float, bool]:
    """
    Estimate the value of h at which stable A-consensus probability
    crosses the selected probability threshold.

    Linear interpolation is used between consecutive tested h values.

    Returns:
        estimated_h:
            Estimated crossing point, capped at the largest tested h.

        right_censored:
            True when the probability is still above the threshold at
            the largest tested h. In that case, the actual crossing
            occurs after the tested range.
    """
    group = group.sort_values("h")

    h_values = group["h"].to_numpy(dtype=float)
    probabilities = group[
        "stable_a"
    ].to_numpy(dtype=float)

    if probabilities[0] < probability_threshold:
        return 0.0, False

    for index in range(1, len(h_values)):
        previous_probability = probabilities[index - 1]
        current_probability = probabilities[index]

        if current_probability < probability_threshold:
            previous_h = h_values[index - 1]
            current_h = h_values[index]

            if np.isclose(
                previous_probability,
                current_probability,
            ):
                return float(current_h), False

            fraction = (
                probability_threshold
                - previous_probability
            ) / (
                current_probability
                - previous_probability
            )

            estimated_h = (
                previous_h
                + fraction * (current_h - previous_h)
            )

            return float(estimated_h), False

    return float(h_values[-1]), True


def build_threshold_summary(
    data: pd.DataFrame,
    probability_threshold: float,
) -> pd.DataFrame:
    rows = []

    group_columns = [
        "model",
        "N",
        "disruptor_count",
        "disruptor_fraction",
    ]

    for key, group in data.groupby(group_columns):
        (
            model,
            N,
            disruptor_count,
            disruptor_fraction,
        ) = key

        estimated_h, censored = estimate_h_threshold(
            group,
            probability_threshold,
        )

        group = group.sort_values("h")

        row = {
            "model": model,
            "N": int(N),
            "disruptor_count": int(disruptor_count),
            "disruptor_fraction": disruptor_fraction,
            "probability_threshold":
                probability_threshold,
            "estimated_h_threshold": estimated_h,
            "right_censored": censored,
            "maximum_tested_h": group["h"].max(),
            "probability_h0":
                group.iloc[0]["stable_a"],
            "probability_hmax":
                group.iloc[-1]["stable_a"],
        }

        reference_rows = group[
            np.isclose(group["h"], 40)
        ]

        row["probability_h40"] = (
            reference_rows.iloc[0][
                "stable_a"
            ]
            if not reference_rows.empty
            else np.nan
        )

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        [
            "model",
            "N",
            "disruptor_fraction",
        ]
    )


def plot_threshold_summary(
    summary: pd.DataFrame,
    output_dir: Path,
    probability_threshold: float,
    dpi: int,
) -> None:
    """
    Compact summary: estimated holding time at which probability
    falls below the selected threshold.
    """
    N_values = sorted(summary["N"].unique())

    colours = plt.cm.viridis(
        np.linspace(
            0.10,
            0.90,
            len(N_values),
        )
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11.5, 7.5),
        sharex=True,
        sharey=True,
    )

    handles = []

    for axis, model in zip(
        axes.flat,
        MODEL_ORDER,
    ):
        model_data = summary[
            summary["model"] == model
        ]

        for colour, N in zip(
            colours,
            N_values,
        ):
            subset = model_data[
                model_data["N"] == N
            ].sort_values("disruptor_fraction")

            if subset.empty:
                continue

            is_censored = subset["right_censored"].astype(bool)

            uncensored = subset[~is_censored]
            censored = subset[is_censored]

            line = axis.plot(
                subset["disruptor_fraction"],
                subset["estimated_h_threshold"],
                # marker="o",
                linewidth=1.8,
                color=colour,
                label=f"N={int(N)}",
            )[0]

            if len(handles) < len(N_values):
                handles.append(line)

            # Ordinary threshold estimates.
            axis.scatter(
                uncensored["disruptor_fraction"],
                uncensored["estimated_h_threshold"],
                marker="o",
                s=30,
                facecolors=colour,
                edgecolors=colour,
                zorder=3,
            )

            # Right-censored estimates.
            axis.scatter(
                censored["disruptor_fraction"],
                censored["estimated_h_threshold"],
                marker="^",
                s=75,
                facecolors=axis.get_facecolor(),
                edgecolors=colour,
                linewidths=2,
                zorder=4,
            )
            axis.set_xticks(np.arange(0.0, 0.61, 0.1))
            axis.set_xticks(np.arange(0.05, 0.61, 0.1), minor=True)
            axis.grid(axis="x",which="minor",alpha=0.20)
            axis.tick_params(axis="x",which="minor",labelbottom=False)

        axis.set_title(
            MODEL_LABELS[model],
        )

        axis.set_xlim(-0.01, 0.61)
        axis.grid(alpha=0.25)

    maximum_h = summary["maximum_tested_h"].max()

    figure.suptitle(
        (
            "Estimated 50% persistence horizon for $A$-consensus"
            if np.isclose(probability_threshold, 0.5)
            else
            "Estimated persistence horizon for $A$-consensus"
        ),
        y=0.98,
    )

    figure.supxlabel(
        "Disruptor fraction",
        y=0.025,
    )

    figure.supylabel(
        (
            r"Estimated $h$ at which "
            r"$P(\mathrm{stable\ A})="
            f"{probability_threshold:.2f}$"
        ),
        x=0.02,
    )

    if handles:
        figure.legend(
            handles=handles,
            labels=[
                f"N={int(N)}"
                for N in N_values
            ],
            loc="upper center",
            ncol=len(N_values),
            bbox_to_anchor=(0.5, 0.94),
        )

    figure.text(
        0.5,
        0.005,
        (
            "Open triangles indicate that the probability remains "
            f"at least {probability_threshold:.2f} at "
            f"the maximum tested holding time h={maximum_h:g}."
        ),
        ha="center",
        fontsize=12,
    )

    figure.subplots_adjust(
        left=0.09,
        right=0.98,
        bottom=0.12,
        top=0.86,
        wspace=0.08,
        hspace=0.22,
    )

    save_figure(
        figure,
        output_dir,
        "smc_h_a_persistence_threshold",
        dpi,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the main plots for the SMC holding-time sweep."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/smc_h_sweep.csv"),
        help="CSV generated by smc_h_sweep.py.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/plots/smc_h"),
        help="Directory in which the figures are saved.",
    )

    parser.add_argument(
        "--reference-h",
        type=float,
        default=40.0,
        help=(
            "Reference holding time shown as a vertical dotted line."
        ),
    )

    parser.add_argument(
        "--probability-threshold",
        type=float,
        default=0.5,
        help=(
            "Probability level used to estimate the persistence horizon."
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution of PNG outputs.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not 0.0 < args.probability_threshold < 1.0:
        raise ValueError(
            "probability-threshold must be between 0 and 1"
        )

    data = load_results(args.input)
    N_values = sorted(map(int, data["N"].unique()))
    h_values = sorted(map(int, data["h"].unique()))

    print(f"Loaded {len(data)} successful rows")
    print(f"N values: {N_values}")
    print(f"h values: {h_values}")

    for N in sorted(data["N"].unique()):
        plot_stability_over_h(
            data=data,
            output_dir=args.output_dir,
            N=int(N),
            reference_h=args.reference_h,
            dpi=args.dpi,
        )

    threshold_summary = build_threshold_summary(
        data,
        args.probability_threshold,
    )

    summary_path = (
        args.output_dir
        / "smc_h_a_persistence_threshold.csv"
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    threshold_summary.to_csv(
        summary_path,
        index=False,
    )

    print(f"Saved: {summary_path}")

    plot_threshold_summary(
        summary=threshold_summary,
        output_dir=args.output_dir,
        probability_threshold=args.probability_threshold,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
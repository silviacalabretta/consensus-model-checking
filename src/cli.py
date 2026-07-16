from __future__ import annotations

import argparse
from pathlib import Path

ALL_MODELS = (
    "voter_zealots",
    "voter_contrarians",
    "crossinh_zealots",
    "crossinh_contrarians",
)


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model", type=str, required=True,
        choices=list(ALL_MODELS),
    )
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--Z", type=int, default=None,
                        help="Total zealots (even). Sets Za=Zb=Z/2.")
    parser.add_argument("--Za", type=int, default=None)
    parser.add_argument("--Zb", type=int, default=None)
    parser.add_argument("--C", type=int, default=4)
    parser.add_argument("--t", type=int, default=35)
    parser.add_argument("--h", type=int, default=40)
    parser.add_argument("--qa", type=float, default=1.05,
                        help="Rate multiplier for A-side interactions")
    parser.add_argument("--qb", type=float, default=0.95,
                        help="Rate multiplier for B-side interactions")


def resolve_model_args(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve --Z into Za/Zb and apply defaults after argparse."""
    if args.Z is not None:
        if args.Za is not None or args.Zb is not None:
            raise ValueError("--Z cannot be combined with --Za or --Zb")
        if args.Z % 2 != 0:
            raise ValueError(f"--Z must be even, got {args.Z}")
        args.Za = args.Z // 2
        args.Zb = args.Z // 2
    else:
        if args.Za is not None and args.Zb is None:
            raise ValueError("--Zb is required when --Za is given")
        if args.Zb is not None and args.Za is None:
            raise ValueError("--Za is required when --Zb is given")
        if args.Za is None and args.Zb is None:
            args.Za = 2
            args.Zb = 2
    return args


def add_simulation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-time", type=float, default=100,
                        help="Max simulation time")


def add_experiment_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Number of simulated trajectories")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.99,
                        help="Confidence level for Wilson intervals")
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable the progress bar")


def add_plot_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input", type=Path,
        default=Path("results/model_check_grid.csv"),
        help="CSV produced by exact_sweep.py",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/plots/model_check_grid"),
        help="Directory in which figures are saved",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display figures interactively after saving them",
    )


def build_model_check_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model_check.py",
        description="Run CTMC model checking on consensus models.",
    )
    add_model_args(parser)
    return parser


def build_simulate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simulate.py",
        description="Simulate a CTMC consensus model.",
    )
    add_model_args(parser)
    add_simulation_args(parser)
    return parser


def build_experiment_parser(
    description: str,
    default_episodes: int = 1000,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    add_model_args(parser)
    parser.set_defaults(episodes=default_episodes)
    add_experiment_args(parser)
    return parser


def build_plot_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    add_plot_args(parser)
    return parser

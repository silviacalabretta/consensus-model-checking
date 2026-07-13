from __future__ import annotations

import argparse


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["voter_zealots", "voter_contrarians",
                 "crossinh_zealots", "crossinh_contrarians"],
    )
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--Za", type=int, default=2)
    parser.add_argument("--Zb", type=int, default=2)
    parser.add_argument("--C", type=int, default=4)
    parser.add_argument("--t", type=int, default=35)
    parser.add_argument("--h", type=int, default=40)


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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-time", type=float, default=100,
                        help="Max simulation time")
    return parser

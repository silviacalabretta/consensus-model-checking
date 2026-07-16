#!/usr/bin/env python3

from src.cli import build_model_check_parser, resolve_model_args
from src.checking import run_model_check
from src.types import ModelParams


def main() -> None:
    args = build_model_check_parser().parse_args()
    args = resolve_model_args(args)
    params = ModelParams(
        model_name=args.model,
        N=args.N,
        Za=args.Za,
        Zb=args.Zb,
        C=args.C,
        t=args.t,
        h=args.h,
        qa=args.qa,
        qb=args.qb,
    )
    result = run_model_check(params)
    print(result)


if __name__ == "__main__":
    main()

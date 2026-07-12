#!/usr/bin/env python3
import sys
import time

from util.cli_args import build_model_check_parser
from util.model_builder import (
    build_ctmc,
    parse_properties_file,
    check_property,
    get_initial_state,
)


def run_model_check(
    model_name: str,
    N: int, Za: int, Zb: int, C: int,
    t: int, h: int,
) -> None:
    print(f"Building CTMC: {model_name}")
    print(f"  N={N}, Za={Za}, Zb={Zb}, C={C}, t={t}, h={h}")
    t0 = time.time()
    bm = build_ctmc(model_name, N=N, Za=Za, Zb=Zb, C=C, t=t, h=h)
    elapsed = time.time() - t0
    print(f"  Built in {elapsed:.3f}s — {bm.model.nr_states} states, "
          f"{bm.model.nr_transitions} transitions\n")

    init = get_initial_state(bm.model)
    print(f"Initial state: {init}")
    print(f"Labels: {sorted(bm.model.labeling.get_labels_of_state(init))}\n")

    print("=" * 60)
    print("PROBABILITY PROPERTIES")
    print("=" * 60)
    for prop in parse_properties_file("prob", bm.prism_program):
        t1 = time.time()
        result = check_property(bm.model, prop)
        val = result.at(init)
        print(f"  {prop}  =>  {val}  ({time.time()-t1:.3f}s)")

    print(f"\n{'='*60}")
    print("EXPECTED TIME PROPERTIES")
    print("=" * 60)
    for prop in parse_properties_file("time", bm.prism_program):
        t1 = time.time()
        result = check_property(bm.model, prop)
        val = result.at(init)
        print(f"  {prop}  =>  {val}  ({time.time()-t1:.3f}s)")


def main():
    args = build_model_check_parser().parse_args()
    run_model_check(args.model, args.N, args.Za, args.Zb, args.C, args.t, args.h)


if __name__ == "__main__":
    main()

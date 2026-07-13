#!/usr/bin/env python3

import time
from dataclasses import dataclass

from util.cli_args import build_model_check_parser
from util.model_builder import (
    build_ctmc,
    parse_properties_file,
    check_property,
    get_initial_state,
)


@dataclass
class PropertyResult:
    formula: str
    value: float
    elapsed_time: float


@dataclass
class ModelCheckResult:
    model_name: str
    N: int
    Za: int
    Zb: int
    C: int
    t: int
    h: int

    build_time: float
    nr_states: int
    nr_transitions: int

    initial_state: int
    initial_labels: list[str]

    probability_results: list[PropertyResult]
    expected_time_results: list[PropertyResult]

    def __str__(self) -> str:
        lines = [
            f"Model: {self.model_name}",
            (
                f"Parameters: N={self.N}, Za={self.Za}, Zb={self.Zb}, "
                f"C={self.C}, t={self.t}, h={self.h}"
            ),
            (
                f"Built in {self.build_time:.3f}s — "
                f"{self.nr_states} states, "
                f"{self.nr_transitions} transitions"
            ),
            "",
            f"Initial state: {self.initial_state}",
            f"Labels: {self.initial_labels}",
            "",
            "=" * 60,
            "PROBABILITY PROPERTIES",
            "=" * 60,
        ]

        for result in self.probability_results:
            lines.append(
                f"  {result.formula}  =>  {result.value}  "
                f"({result.elapsed_time:.3f}s)"
            )

        lines.extend([
            "",
            "=" * 60,
            "EXPECTED TIME PROPERTIES",
            "=" * 60,
        ])

        for result in self.expected_time_results:
            lines.append(
                f"  {result.formula}  =>  {result.value}  "
                f"({result.elapsed_time:.3f}s)"
            )

        return "\n".join(lines)


def check_properties(
    property_type: str,
    model,
    prism_program,
    initial_state: int,
) -> list[PropertyResult]:
    results = []

    for prop in parse_properties_file(property_type, prism_program):
        start_time = time.perf_counter()

        result = check_property(model, prop)
        value = result.at(initial_state)

        elapsed_time = time.perf_counter() - start_time

        results.append(
            PropertyResult(
                formula=str(prop.raw_formula),
                value=float(value),
                elapsed_time=elapsed_time,
            )
        )

    return results


def run_model_check(
    model_name: str,
    N: int,
    Za: int,
    Zb: int,
    C: int,
    t: int,
    h: int,
) -> ModelCheckResult:
    
    # build the model
    build_start = time.perf_counter()
    built_model = build_ctmc(model_name,N=N,Za=Za,Zb=Zb,C=C,t=t,h=h)
    build_time = time.perf_counter() - build_start

    model = built_model.model
    prism_program = built_model.prism_program

    initial_state = get_initial_state(model)
    initial_labels = sorted(model.labeling.get_labels_of_state(initial_state))

    probability_results = check_properties(
        property_type="prob",
        model=model,
        prism_program=prism_program,
        initial_state=initial_state,
    )

    expected_time_results = check_properties(
        property_type="time",
        model=model,
        prism_program=prism_program,
        initial_state=initial_state,
    )

    return ModelCheckResult(
        model_name=model_name,
        N=N,
        Za=Za,
        Zb=Zb,
        C=C,
        t=t,
        h=h,
        build_time=build_time,
        nr_states=model.nr_states,
        nr_transitions=model.nr_transitions,
        initial_state=initial_state,
        initial_labels=initial_labels,
        probability_results=probability_results,
        expected_time_results=expected_time_results,
    )


def main() -> None:
    args = build_model_check_parser().parse_args()

    result = run_model_check(
        model_name=args.model,
        N=args.N,
        Za=args.Za,
        Zb=args.Zb,
        C=args.C,
        t=args.t,
        h=args.h,
    )

    print(result)


if __name__ == "__main__":
    main()
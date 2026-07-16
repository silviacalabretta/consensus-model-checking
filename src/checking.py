from __future__ import annotations

import time
import stormpy

from src.types import ModelCheckResult, ModelParams, PropertyResult
from src.model import build_ctmc, get_initial_state, parse_properties_file


def check_properties(
    property_type: str,
    model,
    prism_program,
    initial_state: int,
) -> list[PropertyResult]:
    results = []

    for prop in parse_properties_file(property_type, prism_program):
        start_time = time.perf_counter()

        result = stormpy.model_checking(model, prop)
        value = result.at(initial_state)

        elapsed_time = time.perf_counter() - start_time
        prop_name = prop.name if prop.name else str(prop.raw_formula)

        results.append(
            PropertyResult(
                name=prop_name,
                formula=str(prop.raw_formula),
                value=float(value),
                elapsed_time=elapsed_time,
            )
        )

    return results


def run_model_check(params: ModelParams) -> ModelCheckResult:
    build_start = time.perf_counter()
    built_model = build_ctmc(params)
    build_time = time.perf_counter() - build_start

    model = built_model.model
    prism_program = built_model.prism_program

    initial_state = get_initial_state(model)
    initial_labels = sorted(model.labeling.get_labels_of_state(initial_state))

    check_start = time.perf_counter()

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

    check_time = time.perf_counter() - check_start

    return ModelCheckResult(
        params=params,
        build_time=build_time,
        check_time=check_time,
        nr_states=model.nr_states,
        nr_transitions=model.nr_transitions,
        initial_state=initial_state,
        initial_labels=initial_labels,
        probability_results=probability_results,
        expected_time_results=expected_time_results,
    )

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List

import stormpy

from src.types import (
    BuiltModel,
    ModelParams,
    SimulationContext,
)

stormpy.set_loglevel_error()

ALL_MODELS = (
    "voter_zealots",
    "voter_contrarians",
    "crossinh_zealots",
    "crossinh_contrarians",
)

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_PROPERTIES_DIR = Path(__file__).resolve().parent.parent / "properties"

_MODEL_FILENAMES = {
    "voter_zealots": "voter_zealots.prism",
    "voter_contrarians": "voter_contrarians.prism",
    "crossinh_zealots": "crossinh_zealots.prism",
    "crossinh_contrarians": "crossinh_contrarians.prism",
}

_PROPS_FILES = {
    "prob": "consensus_prob.props",
    "time": "consensus_time.props",
}

_MODEL_VARIABLES = {
    "voter_zealots": ["a", "b"],
    "voter_contrarians": ["a", "b", "Ca", "Cb"],
    "crossinh_zealots": ["a", "b", "u"],
    "crossinh_contrarians": ["a", "b", "u", "Ca", "Cb"],
}


def _substitute_constants(prism_text: str, params: dict[str, int | float]) -> str:
    for name, value in params.items():
        pattern = (
            rf"(?m)^\s*const\s+(int|double)\s+{re.escape(name)}"
            rf"\s*(?:=\s*[^;]+)?\s*;"
        )
        match = re.search(pattern, prism_text)
        if not match:
            raise ValueError(
                f"Expected exactly one declaration of {name!r}, found 0"
            )
        const_type = match.group(1)
        replacement = f"const {const_type} {name} = {value};"
        prism_text = prism_text[:match.start()] + replacement + prism_text[match.end():]
    return prism_text


def _validate_model_params(params: ModelParams) -> None:
    if params.N <= 0:
        raise ValueError("N must be positive")
    if params.t <= 0:
        raise ValueError("t must be positive")
    if params.h <= 0:
        raise ValueError("h must be positive")
    if "zealot" in params.model_name:
        if params.Za < 0 or params.Zb < 0:
            raise ValueError("Za/Zb must be non-negative")
        a_half = params.N // 2
        b_half = params.N - a_half
        if params.Za > a_half:
            raise ValueError(f"Za must be <= {a_half} for N={params.N}")
        if params.Zb > b_half:
            raise ValueError(f"Zb must be <= {b_half} for N={params.N}")
    if "contrarian" in params.model_name:
        if params.C < 0:
            raise ValueError("C must be non-negative")
        if params.C > params.N:
            raise ValueError(f"C must be <= N ({params.N})")


def build_ctmc(params: ModelParams) -> BuiltModel:
    fname = _MODEL_FILENAMES.get(params.model_name)
    if fname is None:
        raise ValueError(f"Unknown model: {params.model_name}")

    _validate_model_params(params)

    prism_path = _MODELS_DIR / fname
    if not prism_path.exists():
        raise FileNotFoundError(f"PRISM file not found: {prism_path}")

    with open(prism_path, "r") as f:
        prism_text = f.read()

    sub_params: Dict[str, int | float] = {
        "N": params.N, "t": params.t, "h": params.h,
        "qa": params.qa, "qb": params.qb,
    }
    if "zealot" in params.model_name:
        sub_params["Za"] = params.Za
        sub_params["Zb"] = params.Zb
    if "contrarian" in params.model_name:
        sub_params["C"] = params.C

    prism_text = _substitute_constants(prism_text, sub_params)

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".prism", delete=False)
    try:
        tmp.write(prism_text)
        tmp.close()

        prism_program = stormpy.parse_prism_program(tmp.name, prism_compat=True)

        options = stormpy.BuilderOptions()
        options.set_build_state_valuations(True)
        options.set_build_all_labels(True)
        options.set_build_all_reward_models(True)

        model = stormpy.build_sparse_model_with_options(prism_program, options)
        if model.model_type != stormpy.ModelType.CTMC:
            raise TypeError(
                f"Expected a CTMC, but {params.model_name!r} produced {model.model_type}"
            )
    finally:
        os.unlink(tmp.name)

    return BuiltModel(prism_program=prism_program, model=model)


def build_simulation_context(params: ModelParams) -> SimulationContext:
    built_model = build_ctmc(params)
    model = built_model.model
    var_names, var_values = get_state_variables(model, built_model.prism_program, params.model_name)
    return SimulationContext(
        built_model=built_model,
        transitions=get_transitions(model),
        exit_rates=get_exit_rates(model),
        var_names=var_names,
        var_values=var_values,
        initial_state=get_initial_state(model),
    )


def parse_properties_file(prop_key: str, prism_program: object) -> list:
    fname = _PROPS_FILES.get(prop_key)
    if fname is None:
        raise ValueError(f"Unknown properties key: {prop_key}")
    fpath = _PROPERTIES_DIR / fname
    if not fpath.exists():
        raise FileNotFoundError(f"Properties file not found: {fpath}")

    with open(fpath, "r", encoding="utf-8") as file:
        properties_string = file.read()

    try:
        return stormpy.parse_properties_for_prism_program(
            properties_string, prism_program,
        )
    except RuntimeError as exc:
        raise RuntimeError(f"Could not parse properties in {fpath}:\n{exc}") from exc


def get_exit_rates(model: object) -> List[float]:
    n = model.nr_states
    tm = model.transition_matrix
    rates = []
    for s in range(n):
        row_sum = sum(float(entry.value()) for entry in tm.get_row(s))
        rates.append(row_sum)
    return rates


def get_transitions(model: object) -> Dict[int, List[tuple]]:
    tm = model.transition_matrix
    transitions = {}
    for s in range(model.nr_states):
        targets = []
        for entry in tm.get_row(s):
            targets.append((entry.column, float(entry.value())))
        transitions[s] = targets
    return transitions


def get_state_variables(model: object, prism_program: object, model_name: str):
    if model_name not in _MODEL_VARIABLES:
        raise ValueError(f"Unknown model: {model_name}")

    var_names = _MODEL_VARIABLES[model_name]
    variables = {v.name: v for v in prism_program.variables if v.name in var_names}

    missing = [name for name in var_names if name not in variables]
    if missing:
        raise ValueError(f"Variables not found in {model_name}: {missing}")

    values = {}
    for name, var in variables.items():
        values[name] = [
            model.state_valuations.get_value(s, var)
            for s in range(model.nr_states)
        ]
    return var_names, values


def get_initial_state(model: object) -> int:
    initial_states = list(model.initial_states)

    if len(initial_states) != 1:
        raise ValueError(
            "The simulator currently requires exactly one initial state; "
            f"the model has {len(initial_states)}"
        )

    return initial_states[0]

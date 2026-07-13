from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import stormpy

stormpy.set_loglevel_error()


@dataclass
class BuiltModel:
    """Convenience wrapper for a PRISM program and its Storm model."""
    prism_program: object
    model: object


MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PROPERTIES_DIR = Path(__file__).resolve().parent.parent / "properties"

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


def _substitute_constants(prism_text: str, params: Dict[str, int]) -> str:
    for name, value in params.items():
        pattern = rf"(const\s+int\s+{name}\s*(?:=\s*[^;]+)?)\s*;"
        replacement = f"const int {name} = {value};"
        prism_text = re.sub(pattern, replacement, prism_text)
    return prism_text


def _validate_model_params(
    model_name: str, N: int, Za: int, Zb: int, C: int, t: int, h: int,
) -> None:
    if N <= 0:
        raise ValueError("N must be positive")
    if t <= 0:
        raise ValueError("t must be positive")
    if h <= 0:
        raise ValueError("h must be positive")
    if "zealot" in model_name:
        if Za < 0 or Zb < 0:
            raise ValueError("Za/Zb must be non-negative")
        if Za > N // 2 or Zb > N // 2:
            raise ValueError(f"Za/Zb must be <= floor(N/2) for valid initial state (N={N})")
    if "contrarian" in model_name:
        if C < 0:
            raise ValueError("C must be non-negative")
        if C > N:
            raise ValueError(f"C must be <= N ({N})")


def build_ctmc(
    model_name: str,
    N: int = 20,
    Za: int = 2,
    Zb: int = 2,
    C: int = 4,
    t: int = 35,
    h: int = 40,
) -> BuiltModel:
    fname = _MODEL_FILENAMES.get(model_name)
    if fname is None:
        raise ValueError(f"Unknown model: {model_name}")

    _validate_model_params(model_name, N, Za, Zb, C, t, h)

    prism_path = MODELS_DIR / fname
    if not prism_path.exists():
        raise FileNotFoundError(f"PRISM file not found: {prism_path}")

    with open(prism_path, "r") as f:
        prism_text = f.read()

    params: Dict[str, int] = {"N": N, "t": t, "h": h}
    if "zealot" in model_name:
        params["Za"] = Za
        params["Zb"] = Zb
    if "contrarian" in model_name:
        params["C"] = C

    prism_text = _substitute_constants(prism_text, params)

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
                f"Expected a CTMC, but {model_name!r} produced {model.model_type}")
    finally:
        os.unlink(tmp.name)

    return BuiltModel(prism_program=prism_program, model=model)


def parse_properties_file(prop_key: str, prism_program: object) -> list:
    fname = _PROPS_FILES.get(prop_key)
    if fname is None:
        raise ValueError(f"Unknown properties key: {prop_key}")
    fpath = PROPERTIES_DIR / fname
    if not fpath.exists():
        raise FileNotFoundError(f"Properties file not found: {fpath}")

    properties = []

    with open(fpath, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            formula = line.strip()

            # Ignore empty lines and comment lines
            if not formula or formula.startswith("//"):
                continue

            # A trailing semicolon is optional when parsing individually,
            # so remove it for consistency.
            formula = formula.removesuffix(";").strip()

            try:
                parsed = stormpy.parse_properties_for_prism_program(
                    formula,
                    prism_program,
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Could not parse property in {fpath}, "
                    f"line {line_number}:\n{formula}"
                ) from exc

            properties.extend(parsed)

    return properties


def check_property(model: object, property_obj: object) -> object:
    return stormpy.model_checking(model, property_obj)


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


_MODEL_VARIABLES = {
    "voter_zealots": ["a", "b"],
    "voter_contrarians": ["a", "b", "Ca", "Cb"],
    "crossinh_zealots": ["a", "b", "u"],
    "crossinh_contrarians": ["a", "b", "u", "Ca", "Cb"],
}


def get_state_variables(model: object, prism_program: object, model_name: str):
    if model_name not in _MODEL_VARIABLES:
        raise ValueError(f"Unknown model: {model_name}")
    
    var_names = _MODEL_VARIABLES.get(model_name, [])
    variables = {v.name: v for v in prism_program.variables if v.name in var_names}
    
    missing = [name for name in var_names if name not in variables]
    if missing:
        raise ValueError(f"Variables not found in {model_name}: {missing}")
    
    values = {}
    for name, var in variables.items():
        values[name] = [model.state_valuations.get_value(s, var)
                        for s in range(model.nr_states)]
    return var_names, values


def get_initial_state(model: object) -> int:
    initial_states = list(model.initial_states)

    if len(initial_states) != 1:
        raise ValueError(
            "The simulator currently requires exactly one initial state; "
            f"the model has {len(initial_states)}"
        )

    return initial_states[0]

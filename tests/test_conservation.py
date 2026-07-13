import pytest

from util.model_builder import build_ctmc, get_state_variables


ZEALOT_MODELS = [
    "voter_zealots",
    "crossinh_zealots",
]

CONTRARIAN_MODELS = [
    "voter_contrarians",
    "crossinh_contrarians",
]

ALL_MODELS = ZEALOT_MODELS + CONTRARIAN_MODELS


@pytest.mark.parametrize("model_name", ZEALOT_MODELS)
def test_zealot_population_is_conserved(model_name: str) -> None:
    N = 6
    Za = 1
    Zb = 1

    built_model = build_ctmc(
        model_name=model_name,
        N=N,
        Za=Za,
        Zb=Zb,
        C=0,
        t=5,
        h=3,
    )

    model = built_model.model

    _, values = get_state_variables(
        model,
        built_model.prism_program,
        model_name,
    )

    for state in range(model.nr_states):
        total = (
            int(values["a"][state])
            + int(values["b"][state])
            + Za
            + Zb
        )

        if "u" in values:
            total += int(values["u"][state])

        assert total == N, (
            f"Population conservation violated in "
            f"{model_name}, state {state}: "
            f"expected {N}, obtained {total}"
        )


@pytest.mark.parametrize("model_name", CONTRARIAN_MODELS)
def test_contrarian_population_is_conserved(
    model_name: str,
) -> None:
    N = 6
    C = 2

    built_model = build_ctmc(
        model_name=model_name,
        N=N,
        Za=0,
        Zb=0,
        C=C,
        t=5,
        h=3,
    )

    model = built_model.model

    _, values = get_state_variables(
        model,
        built_model.prism_program,
        model_name,
    )

    for state in range(model.nr_states):
        ca = int(values["Ca"][state])
        cb = int(values["Cb"][state])

        total = (
            int(values["a"][state])
            + int(values["b"][state])
            + ca
            + cb
        )

        if "u" in values:
            total += int(values["u"][state])

        assert total == N, (
            f"Population conservation violated in "
            f"{model_name}, state {state}: "
            f"expected {N}, obtained {total}"
        )

        assert ca + cb == C, (
            f"Contrarian conservation violated in "
            f"{model_name}, state {state}: "
            f"expected {C}, obtained {ca + cb}"
        )


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_majority_labels_are_mutually_exclusive(
    model_name: str,
) -> None:
    built_model = build_ctmc(
        model_name=model_name,
        N=6,
        Za=1,
        Zb=1,
        C=2,
        t=5,
        h=3,
    )

    model = built_model.model

    for state in range(model.nr_states):
        labels = set(
            model.labeling.get_labels_of_state(state)
        )

        assert not (
            "maj_a" in labels
            and "maj_b" in labels
        ), (
            f"State {state} in {model_name} is labelled "
            "both maj_a and maj_b"
        )


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_transition_rates_are_non_negative(
    model_name: str,
) -> None:
    built_model = build_ctmc(
        model_name=model_name,
        N=6,
        Za=1,
        Zb=1,
        C=2,
        t=5,
        h=3,
    )

    model = built_model.model
    transition_matrix = model.transition_matrix

    for state in range(model.nr_states):
        for entry in transition_matrix.get_row(state):
            rate = float(entry.value())

            assert rate >= 0.0, (
                f"Negative rate {rate} in "
                f"{model_name}, state {state}"
            )
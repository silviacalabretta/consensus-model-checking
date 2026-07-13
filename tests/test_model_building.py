import pytest
import stormpy

from util.model_builder import build_ctmc, get_initial_state


MODEL_NAMES = [
    "voter_zealots",
    "voter_contrarians",
    "crossinh_zealots",
    "crossinh_contrarians",
]


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_model_builds_as_ctmc(model_name: str) -> None:
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

    assert model.model_type == stormpy.ModelType.CTMC
    assert model.nr_states > 0
    assert model.nr_transitions >= 0


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_model_has_one_initial_state(model_name: str) -> None:
    built_model = build_ctmc(
        model_name=model_name,
        N=6,
        Za=1,
        Zb=1,
        C=2,
        t=5,
        h=3,
    )

    initial_state = get_initial_state(built_model.model)

    assert isinstance(initial_state, int)
    assert 0 <= initial_state < built_model.model.nr_states


def test_invalid_population_size_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_ctmc(
            model_name="voter_zealots",
            N=0,
            Za=0,
            Zb=0,
            C=0,
            t=5,
            h=3,
        )


def test_unknown_model_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_ctmc(
            model_name="unknown_model",
            N=6,
            Za=1,
            Zb=1,
            C=2,
            t=5,
            h=3,
        )
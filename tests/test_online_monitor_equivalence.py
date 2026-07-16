import random

import pytest

from src.gillespie_py import simulate_direct_episode, simulate_direct_with_monitors
from src.monitors import reaches_and_holds, reaches_label_before
from src.types import ModelParams


TEST_CONFIGS = [
    {"model_name": "voter_zealots", "N": 6, "Za": 1, "Zb": 1, "C": 0, "t": 5, "h": 3},
    {"model_name": "voter_zealots", "N": 20, "Za": 2, "Zb": 2, "C": 0, "t": 10, "h": 10},
    {"model_name": "voter_contrarians", "N": 6, "Za": 0, "Zb": 0, "C": 2, "t": 5, "h": 3},
    {"model_name": "voter_contrarians", "N": 20, "Za": 0, "Zb": 0, "C": 4, "t": 10, "h": 10},
    {"model_name": "crossinh_zealots", "N": 6, "Za": 1, "Zb": 1, "C": 0, "t": 5, "h": 3},
    {"model_name": "crossinh_zealots", "N": 20, "Za": 2, "Zb": 2, "C": 0, "t": 10, "h": 10},
    {"model_name": "crossinh_contrarians", "N": 6, "Za": 0, "Zb": 0, "C": 2, "t": 5, "h": 3},
    {"model_name": "crossinh_contrarians", "N": 20, "Za": 0, "Zb": 0, "C": 4, "t": 10, "h": 10},
]

SEEDS = [0, 42, 123]


def _build_monitors(params: ModelParams):
    t = float(params.t)
    h = float(params.h)
    stability = {
        "stable_a": ("maj_a", t, h),
        "stable_b": ("maj_b", t, h),
    }
    reach = {
        "reach_a": ("maj_a", t),
        "reach_b": ("maj_b", t),
        "reach_consensus": ("consensus", t),
    }
    return stability, reach


def _episode_monitors(episode, stability_monitors, reach_monitors):
    results = {}
    for name, (label, rb, ht) in stability_monitors.items():
        results[name] = reaches_and_holds(episode, label, rb, ht)
    for name, (label, tb) in reach_monitors.items():
        results[name] = reaches_label_before(episode, label, tb)
    return results


@pytest.mark.parametrize("cfg", TEST_CONFIGS, ids=lambda c: c["model_name"])
@pytest.mark.parametrize("seed", SEEDS)
def test_direct_monitors_agree(cfg, seed):
    params = ModelParams(**cfg)
    max_time = params.t + params.h
    stability_monitors, reach_monitors = _build_monitors(params)

    rng_episode = random.Random(seed)
    rng_inline = random.Random(seed)

    ep = simulate_direct_episode(
        params=params,
        rng=rng_episode,
        max_time=max_time,
    )

    inline = simulate_direct_with_monitors(
        params=params,
        rng=rng_inline,
        max_time=max_time,
        stability_monitors=stability_monitors,
        reach_monitors=reach_monitors,
    )

    episode_mon = _episode_monitors(ep, stability_monitors, reach_monitors)

    for name in inline:
        assert inline[name] == episode_mon[name], (
            f"{name}: inline={inline[name]} vs episode={episode_mon[name]} "
            f"({cfg['model_name']}, N={cfg['N']}, seed={seed})"
        )

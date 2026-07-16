from __future__ import annotations

import math
from statistics import NormalDist

from src.types import Estimate


def estimate_probability(outcomes: list[bool]) -> Estimate:
    episodes = len(outcomes)

    if episodes == 0:
        raise ValueError("At least one outcome is required.")

    successes = sum(outcomes)
    probability = successes / episodes
    standard_error = math.sqrt(probability * (1.0 - probability) / episodes)

    return Estimate(
        successes=successes,
        episodes=episodes,
        probability=probability,
        standard_error=standard_error,
    )


def wilson_interval(
    successes: int,
    episodes: int,
    confidence: float,
) -> tuple[float, float]:
    """Wilson confidence interval for a Bernoulli probability."""
    estimate = successes / episodes
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    z_squared = z * z

    denominator = 1.0 + z_squared / episodes
    centre = (
        estimate + z_squared / (2.0 * episodes)
    ) / denominator
    radius = (
        z
        / denominator
        * (
            estimate * (1.0 - estimate) / episodes
            + z_squared / (4.0 * episodes * episodes)
        )
        ** 0.5
    )
    lower = max(0.0, centre - radius)
    upper = min(1.0, centre + radius)

    if successes == 0:
        lower = 0.0
    if successes == episodes:
        upper = 1.0

    return lower, upper

from __future__ import annotations

from src.types import EpisodeResult


def reaches_label_before(
    episode: EpisodeResult,
    label: str,
    time_bound: float,
) -> bool:
    """Check F_[0,time_bound] (label)."""
    return any(step.time <= time_bound and label in step.labels for step in episode.steps)


def reaches_and_holds(
    episode: EpisodeResult,
    label: str,
    reach_bound: float,
    holding_time: float,
) -> bool:
    """Check F_[0,reach_bound] (G_[0,holding_time] (label))."""
    if reach_bound < 0:
        raise ValueError("Monitor: reach_bound must be non-negative")

    if holding_time < 0:
        raise ValueError("Monitor: holding_time must be non-negative")

    sat_start: float | None = None

    for index, step in enumerate(episode.steps):
        step_end = (
            episode.steps[index + 1].time if index + 1 < len(episode.steps)
            else episode.final_time
        )
        if step_end < step.time:
            raise ValueError("Trajectory times must be non-decreasing")

        if label in step.labels:
            if sat_start is None:
                sat_start = step.time

            if (sat_start <= reach_bound and step_end - sat_start >= holding_time):
                return True
        else:
            sat_start = None

    return False

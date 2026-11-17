"""Baseline maintenance policies to compare the PPO agent against.

Each policy is a callable (obs, info, rng) -> action index. Only the PPO agent and the rule-based
policy are restricted to what an engineer can see (obs); periodic maintenance also uses the bearing's
age, which a real plant would know from its maintenance log.
"""
from __future__ import annotations

import numpy as np

from pdm.rl.env import OBS_NAMES

IDX = {n: i for i, n in enumerate(OBS_NAMES)}
CONTINUE, MONITOR, INSPECT, REDUCE, STOP = range(5)


def run_to_failure(obs, info, rng) -> int:
    return CONTINUE


def random_policy(obs, info, rng) -> int:
    return int(rng.integers(0, 5))


def make_periodic(n_days: int):
    def periodic(obs, info, rng) -> int:
        return STOP if info["bearing_age_days"] >= n_days else CONTINUE
    periodic.__name__ = f"periodic_{n_days}"
    return periodic


def rule_based(obs, info=None, rng=None) -> int:
    """What a careful engineer following the health rules would do.

    Healthy  -> continue
    Warning  -> switch on frequent monitoring; if it stays Warning, inspect (at most monthly);
                if the inspection found a fault of degradation >= 0.5, replace
    Critical -> stop for maintenance
    Low confidence -> monitor more frequently (re-measure)
    """
    healthy, warning, critical = obs[IDX["health_Healthy"]], obs[IDX["health_Warning"]], obs[IDX["health_Critical"]]
    monitoring = obs[IDX["monitoring_on"]] > 0.5
    found = obs[IDX["inspection_found_fault"]] > 0.5
    days_since_insp = obs[IDX["days_since_inspection_norm"]] * 180
    if found and obs[IDX["inspected_degradation"]] >= 0.5:
        return STOP
    if critical > 0.5:
        return STOP
    if obs[IDX["low_confidence"]] > 0.5 and not monitoring:
        return MONITOR
    if warning > 0.5:
        if not monitoring:
            return MONITOR
        if days_since_insp > 30:
            return INSPECT
        return CONTINUE
    return CONTINUE

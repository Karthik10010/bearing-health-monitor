"""Run policies on the same simulated episodes and compare them (all results are SIMULATED)."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from pdm.rl.env import ACTIONS, BearingMaintenanceEnv, ObservationModel


def ppo_policy(model) -> Callable:
    def act(obs, info, rng) -> int:
        a, _ = model.predict(obs, deterministic=True)
        return int(a)
    act.__name__ = "ppo"
    return act


def run_episode(env: BearingMaintenanceEnv, policy: Callable, seed: int, record: bool = False) -> dict:
    rng = np.random.default_rng(seed + 10_000)
    obs, info = env.reset(seed=seed)
    total, done, steps = 0.0, False, []
    while not done:
        a = policy(obs, info, rng)
        if record:
            steps.append({"day": info["day"], "degradation": info["degradation"], "true_fault": info["true_fault"],
                          "diagnosis": info["diagnosis"], "confidence": info["confidence"],
                          "health": info["health_observed"], "action": ACTIONS[a]})
        obs, r, term, trunc, info = env.step(a)
        total += r
        done = term or trunc
    days = max(info["day"], 1)
    out = {"seed": seed, "reward": total, "failures": info["failures"], "maintenances": info["maintenances"],
           "unnecessary_maintenances": info["unnecessary_maintenances"], "inspections": info["inspections"],
           "downtime_days": info["downtime_days"], "safety_days": info["safety_days"],
           "availability": info["producing_days"] / env.cfg["episode_days"], "days_survived": days}
    if record:
        out["timeline"] = steps
    return out


def evaluate_policies(policies: dict[str, Callable], n_episodes: int, seed0: int = 1_000_000,
                      obs_model: ObservationModel | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every policy plays the SAME n_episodes (same seeds -> same loads, faults and random wear)."""
    obs_model = obs_model or ObservationModel()
    rows = []
    for name, pol in policies.items():
        env = BearingMaintenanceEnv(obs_model=obs_model)
        for i in range(n_episodes):
            r = run_episode(env, pol, seed0 + i)
            r["policy"] = name
            rows.append(r)
    per_ep = pd.DataFrame(rows)
    summary = per_ep.groupby("policy").agg(
        mean_reward=("reward", "mean"), std_reward=("reward", "std"),
        failures_per_episode=("failures", "mean"), maintenances_per_episode=("maintenances", "mean"),
        unnecessary_maintenance_per_episode=("unnecessary_maintenances", "mean"),
        inspections_per_episode=("inspections", "mean"), downtime_days=("downtime_days", "mean"),
        safety_days=("safety_days", "mean"), availability_pct=("availability", lambda s: 100 * s.mean()),
    ).sort_values("mean_reward", ascending=False)
    summary["sem_reward"] = per_ep.groupby("policy")["reward"].sem()
    return summary, per_ep

"""'What if we keep running?' - SIMULATED forward rollouts from the current diagnosis.

We don't know the bearing's hidden state, only the diagnosis. So each rollout draws a plausible
hidden state from the diagnosis (fault type from the class probabilities, degradation from the
estimated size band) and then runs the simulator with a fixed policy. Results are SIMULATED.
"""
from __future__ import annotations

import numpy as np

from pdm.config import load_config
from pdm.rl.env import FAULTS, BearingMaintenanceEnv, ObservationModel


def _seed_state(env: BearingMaintenanceEnv, probabilities: dict, size_in: float | None, rng) -> None:
    th = [0.0] + list(env.cfg["degradation"]["size_thresholds"]) + [1.0]
    classes = list(probabilities)
    p = np.array([probabilities[c] for c in classes], dtype=float)
    cls = classes[rng.choice(len(classes), p=p / p.sum())]
    if cls == "Normal":
        env.initiated, env.degradation = False, 0.0
    else:
        env.initiated, env.fault = True, cls if cls in FAULTS else env.fault
        sizes = [0.007, 0.014, 0.021, 0.028]
        k = sizes.index(min(sizes, key=lambda s: abs(s - (size_in or 0.007))))
        env.degradation = float(rng.uniform(th[k], th[k + 1] * 0.999))


def simulate_continue(probabilities: dict, size_in: float | None, load_hp: int, n_rollouts: int = 500,
                      horizon_days: int = 180, action: int = 0, seed: int = 0,
                      obs_model: ObservationModel | None = None) -> dict:
    """Run `n_rollouts` simulations doing `action` every day (0 = continue). Returns failure statistics."""
    obs_model = obs_model or ObservationModel()
    cfg = dict(load_config()["rl"])
    cfg["episode_days"] = horizon_days
    env = BearingMaintenanceEnv(cfg=cfg, obs_model=obs_model, fixed_load=int(load_hp))
    rng = np.random.default_rng(seed)
    fail_day = []
    for i in range(n_rollouts):
        env.reset(seed=seed + i)
        _seed_state(env, probabilities, size_in, rng)
        done, failed = False, False
        while not done:
            _, _, term, trunc, info = env.step(action)
            done = term or trunc
            failed = failed or info["failures"] > 0
        fail_day.append(info["day"] if failed else np.inf)
    fd = np.array(fail_day)
    out = {"simulated": True, "n_rollouts": n_rollouts, "horizon_days": horizon_days, "policy": "continue operation",
           "load_hp": int(load_hp)}
    for d in (7, 30, 90, horizon_days):
        out[f"p_failure_within_{d}_days"] = float(np.mean(fd <= d))
    finite = fd[np.isfinite(fd)]
    out["median_days_to_failure_if_fails"] = float(np.median(finite)) if finite.size else None
    return out

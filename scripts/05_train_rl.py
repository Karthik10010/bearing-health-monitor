"""Step 5: train PPO on the SIMULATED maintenance game, one run per seed.

    python scripts/05_train_rl.py              # all seeds from config.yaml (rl.ppo.seeds)
    python scripts/05_train_rl.py --seed 1     # just one seed

Outputs: models/ppo_seed<N>.zip, reports/metrics/ppo_curve_seed<N>.csv
"""
import _bootstrap  # noqa: F401

import argparse
import time

import pandas as pd
import torch

from pdm.config import load_config, path
from pdm.rl.train import train_ppo

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, action="append", help="seed(s) to train; default: config")
    ap.add_argument("--timesteps", type=int, default=None)
    a = ap.parse_args()
    torch.set_num_threads(2)
    seeds = a.seed or load_config()["rl"]["ppo"]["seeds"]
    for s in seeds:
        t = time.time()
        model, curve = train_ppo(seed=s, total_timesteps=a.timesteps)
        model.save(str(path("models_dir") / f"ppo_seed{s}.zip"))
        pd.DataFrame(curve, columns=["timestep", "episode_reward"]).to_csv(
            path("metrics_dir") / f"ppo_curve_seed{s}.csv", index=False)
        last = curve[-200:, 1].mean() if len(curve) else float("nan")
        print(f"seed {s}: {len(curve)} episodes, mean reward of last 200 = {last:.1f}, {time.time() - t:.0f}s")

"""PPO training (Stable-Baselines3) on the simulated maintenance game."""
from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from pdm.config import load_config
from pdm.rl.env import BearingMaintenanceEnv, ObservationModel


class EpisodeRewardLogger(BaseCallback):
    """Records (timestep, episode reward) for the training curve."""

    def __init__(self):
        super().__init__()
        self.records: list[tuple[int, float]] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self.records.append((int(self.num_timesteps), float(ep["r"])))
        return True


def train_ppo(seed: int, total_timesteps: int | None = None, verbose: int = 0):
    cfg = load_config()["rl"]["ppo"]
    total_timesteps = total_timesteps or cfg["total_timesteps"]
    obs_model = ObservationModel()  # shared, read-only
    venv = make_vec_env(lambda: BearingMaintenanceEnv(obs_model=obs_model), n_envs=cfg["n_envs"], seed=seed)
    # Rewards range from +10/day to -300 (breakdown); normalising them keeps PPO's updates stable.
    # Observations are already in [0, 1] and are NOT normalised, so the saved policy can be used directly.
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=cfg["gamma"])
    model = PPO("MlpPolicy", venv, learning_rate=cfg["learning_rate"], n_steps=cfg["n_steps"],
                batch_size=cfg["batch_size"], gamma=cfg["gamma"], ent_coef=cfg["ent_coef"],
                policy_kwargs={"net_arch": [64, 64]}, seed=seed, verbose=verbose, device="cpu")
    cb = EpisodeRewardLogger()
    model.learn(total_timesteps=total_timesteps, callback=cb, progress_bar=False)
    return model, np.array(cb.records)


def action_probabilities(model, obs: np.ndarray) -> np.ndarray:
    """Probability the trained policy assigns to each of the 5 actions for one observation."""
    import torch

    with torch.no_grad():
        t, _ = model.policy.obs_to_tensor(obs.reshape(1, -1))
        return model.policy.get_distribution(t).distribution.probs.cpu().numpy()[0]

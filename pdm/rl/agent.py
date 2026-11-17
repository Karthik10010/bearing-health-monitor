"""Runtime use of the trained PPO agent: diagnosis -> observation -> recommended action.

SAFETY SHIELD (config rl.safety_shield): in simulation the agent learned that most "Critical" readings
are false alarms (size estimates are unreliable on unseen bearings) and often keeps running. That is
not acceptable as advice on a real Critical reading, so when the rating is Critical the agent may not
recommend "Continue operation" or "Monitor more frequently": the rule-based action is used instead.
In simulation the shielded agent still had 0 breakdowns and beat the rule-based policy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from pdm.config import load_config, path
from pdm.rl.baselines import CONTINUE, MONITOR, rule_based
from pdm.rl.env import ACTIONS, OBS_NAMES, Measurement, encode_observation

_CRITICAL = OBS_NAMES.index("health_Critical")


@dataclass
class Recommendation:
    action: str                          # final recommendation (after the safety shield)
    action_index: int
    action_probabilities: dict[str, float]   # the agent's own probabilities
    rule_based_action: str
    agrees_with_rule: bool
    agent_action: str                    # what the agent itself chose
    shielded: bool                       # True if the safety shield replaced the agent's choice
    shield_reason: str | None
    observation: list[float]
    assumptions: list[str]
    label: str = "Trained in a SIMULATION with illustrative costs - not a validated plant policy."

    def to_dict(self) -> dict:
        return asdict(self)


def build_observation(diag: str, conf: float, est_size: float | None, health: str, low_conf: bool, load_hp: int,
                      monitoring_on: bool = False, days_since_inspection: int | None = None,
                      inspected_degradation: float = 0.0, inspection_found_fault: bool = False
                      ) -> tuple[np.ndarray, list[str]]:
    assumptions = []
    if days_since_inspection is None:
        days_since_inspection = load_config()["rl"]["runtime"]["days_since_inspection"]
        assumptions.append(f"days since last inspection assumed = {days_since_inspection} (config.yaml)")
    assumptions.append("frequent monitoring assumed OFF; no recent inspection result")
    load = int(min(3, max(0, round(load_hp))))
    if load != load_hp:
        assumptions.append(f"load {load_hp} HP mapped to the nearest simulated load {load} HP")
    obs = encode_observation(Measurement(diag, conf, est_size, health, low_conf), load, monitoring_on, 0,
                             days_since_inspection, inspected_degradation, inspection_found_fault)
    return obs, assumptions


def shield(obs: np.ndarray, agent_action: int) -> tuple[int, str | None]:
    """Return (final action, reason or None). Applies only when the shield is enabled in config."""
    if not load_config()["rl"].get("safety_shield", True):
        return agent_action, None
    if obs[_CRITICAL] > 0.5 and agent_action in (CONTINUE, MONITOR):
        rule = rule_based(obs)
        return rule, (f"safety shield: the rating is Critical, so '{ACTIONS[agent_action]}' is not allowed; "
                      f"the rule-based action '{ACTIONS[rule]}' is used")
    return agent_action, None


def shielded_policy(policy):
    """Wrap any (obs, info, rng) -> action policy with the safety shield (used in evaluation)."""
    def act(obs, info, rng) -> int:
        return shield(obs, policy(obs, info, rng))[0]
    act.__name__ = f"shielded_{getattr(policy, '__name__', 'policy')}"
    return act


class Agent:
    def __init__(self, model_path=None):
        from stable_baselines3 import PPO

        self.model = PPO.load(str(model_path or path("models_dir") / "ppo_agent.zip"), device="cpu")

    def recommend(self, obs: np.ndarray, assumptions: list[str] | None = None) -> Recommendation:
        from pdm.rl.train import action_probabilities

        probs = action_probabilities(self.model, obs)
        a = int(probs.argmax())  # same as deterministic=True
        final, reason = shield(obs, a)
        r = rule_based(obs)
        return Recommendation(ACTIONS[final], final, {ACTIONS[i]: float(p) for i, p in enumerate(probs)}, ACTIONS[r],
                              final == r, ACTIONS[a], reason is not None, reason, [float(x) for x in obs],
                              assumptions or [])

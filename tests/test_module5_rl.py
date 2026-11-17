"""Module 5 (RL simulator) tests. Needs the OOF files from scripts/03_train_classifier.py."""
import numpy as np
import pytest

from pdm.config import path

pytestmark = pytest.mark.skipif(not (path("metrics_dir") / "fault_oof_unseen_bearing.csv").exists(),
                                reason="run scripts/03_train_classifier.py first")


@pytest.fixture(scope="module")
def obs_model():
    from pdm.rl.env import ObservationModel
    return ObservationModel()


@pytest.fixture
def env(obs_model):
    from pdm.rl.env import BearingMaintenanceEnv
    return BearingMaintenanceEnv(obs_model=obs_model)


def test_gymnasium_and_sb3_env_checkers(env):
    from gymnasium.utils.env_checker import check_env as gym_check
    from stable_baselines3.common.env_checker import check_env as sb3_check
    gym_check(env, skip_render_check=True)
    sb3_check(env, warn=True)


def test_recording_mapping(obs_model):
    assert obs_model.recording_id("Outer race", 0.028, 2) == "OR021@6_2"  # no 0.028" outer race in CWRU
    assert obs_model.recording_id(None, 0.007, 3) == "Normal_3"
    assert obs_model.recording_id("Ball", 0.014, 0) == "B014_0"
    for f in ("Inner race", "Outer race", "Ball"):
        for s in (0.007, 0.014, 0.021, 0.028):
            for load in range(4):
                assert obs_model.recording_id(f, s, load) in obs_model.fault_p


def test_observations_valid_and_episode_ends(env):
    obs, info = env.reset(seed=3)
    for _ in range(400):
        assert env.observation_space.contains(obs)
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        if term or trunc:
            break
    assert term or trunc
    assert info["day"] <= env.cfg["episode_days"] + env.cfg["costs"]["maintenance_downtime_days"]


def test_run_to_failure_often_fails_and_maintenance_resets(env):
    fails = 0
    for s in range(60):
        env.reset(seed=s)
        done = False
        while not done:
            _, _, term, trunc, info = env.step(0)
            done = term or trunc
        fails += info["failures"]
    assert fails >= 15  # wear makes breakdowns common without maintenance

    env.reset(seed=1)
    env.initiated, env.degradation = True, 0.6
    _, r, *_ , info = env.step(4)
    assert info["degradation"] == 0.0 and info["maintenances"] == 1 and info["unnecessary_maintenances"] == 0
    assert r == pytest.approx(-env.cfg["costs"]["maintenance"])


def test_unnecessary_maintenance_penalty(env):
    env.reset(seed=2)
    env.initiated, env.degradation = False, 0.0
    _, r, *_ , info = env.step(4)
    c = env.cfg["costs"]
    assert r == pytest.approx(-(c["maintenance"] + c["unnecessary_maintenance"]))
    assert info["unnecessary_maintenances"] == 1


def test_inspection_reveals_truth(env):
    from pdm.rl.env import OBS_NAMES
    env.reset(seed=5)
    env.initiated, env.degradation = True, 0.42
    obs, *_ = env.step(2)
    assert obs[OBS_NAMES.index("inspection_found_fault")] == 1.0
    assert obs[OBS_NAMES.index("inspected_degradation")] == pytest.approx(0.42)


def test_baselines_and_whatif(obs_model):
    from pdm.rl.baselines import make_periodic, rule_based, run_to_failure
    from pdm.rl.evaluate import evaluate_policies
    from pdm.rl.whatif import simulate_continue
    summary, per_ep = evaluate_policies({"rtf": run_to_failure, "rule": rule_based, "p60": make_periodic(60)},
                                        n_episodes=40, obs_model=obs_model)
    assert set(summary.index) == {"rtf", "rule", "p60"} and len(per_ep) == 120
    assert summary.loc["rtf", "maintenances_per_episode"] == 0
    healthy = simulate_continue({"Normal": 1.0, "Inner race": 0, "Outer race": 0, "Ball": 0}, None, 1,
                                n_rollouts=60, obs_model=obs_model)
    critical = simulate_continue({"Normal": 0, "Inner race": 1.0, "Outer race": 0, "Ball": 0}, 0.028, 3,
                                 n_rollouts=60, obs_model=obs_model)
    assert healthy["simulated"] is True
    assert critical["p_failure_within_30_days"] > healthy["p_failure_within_30_days"]


def test_safety_shield():
    from pdm.rl.agent import build_observation, shield
    from pdm.rl.baselines import CONTINUE, INSPECT, MONITOR, STOP
    crit, _ = build_observation("Outer race", 0.9, 0.021, "Critical", False, 0)
    warn, _ = build_observation("Ball", 0.8, 0.007, "Warning", False, 1)
    for a in (CONTINUE, MONITOR):
        final, reason = shield(crit, a)
        assert final == STOP and "Critical" in reason          # replaced by the rule-based action
    assert shield(crit, INSPECT) == (INSPECT, None)            # other actions are allowed
    assert shield(warn, CONTINUE) == (CONTINUE, None)          # only Critical ratings are shielded

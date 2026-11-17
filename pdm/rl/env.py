"""BearingMaintenanceEnv - a SIMULATED maintenance game for one bearing (Gymnasium).

Plain words: each step is one day. Hidden from the agent, a bearing may develop a fault that grows
a little every day (faster under higher load). The agent only sees what a real engineer would see:
the latest diagnosis from the vibration model (sampled from the REAL Random Forest results on
UNSEEN bearings, so it includes realistic mistakes), the rule-based health status, the load, and
what the last inspection revealed. It picks one of 5 actions and earns production value minus costs;
a breakdown is very expensive and ends the episode.

All costs, wear rates and failure probabilities are ILLUSTRATIVE ASSUMPTIONS from config.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from pdm.config import load_config, path
from pdm.data.cwru_catalog import CLASSES, LOAD_RPM
from pdm.health import HEALTH_LEVELS, assess_health

ACTIONS = ["Continue operation", "Monitor more frequently", "Schedule inspection",
           "Reduce operating load", "Stop for maintenance"]
FAULTS = ["Inner race", "Outer race", "Ball"]
SIZES = [0.007, 0.014, 0.021, 0.028]
OBS_NAMES = ([f"diag_{c}" for c in CLASSES] + ["diag_confidence", "est_size_norm"]
             + [f"health_{h}" for h in HEALTH_LEVELS]
             + ["low_confidence", "load_norm", "monitoring_on", "days_since_measurement_norm",
                "days_since_inspection_norm", "inspected_degradation", "inspection_found_fault"])
OBS_DIM = len(OBS_NAMES)  # 16


# ------------------------------------------------------------------ observation model (real RF results)
class ObservationModel:
    """Samples a 'measurement' for a given true state by averaging K consecutive windows of the
    matching CWRU recording's OUT-OF-FOLD (never-seen-bearing) predicted probabilities."""

    def __init__(self, fault_oof: pd.DataFrame | None = None, sev_oof: pd.DataFrame | None = None):
        m = path("metrics_dir")
        fault_oof = fault_oof if fault_oof is not None else pd.read_csv(m / "fault_oof_unseen_bearing.csv")
        sev_oof = sev_oof if sev_oof is not None else pd.read_csv(m / "severity_oof_unseen_type.csv")
        self.fault_p: dict[str, np.ndarray] = {}
        self.knock: dict[str, np.ndarray] = {}   # envelope peaks per window, for the physics check
        kcols = ["e_bpfo_1", "e_bpfi_1", "e_bsf_1", "e_bsf_2"]
        for rid, g in fault_oof.sort_values(["recording_id", "window_idx"]).groupby("recording_id"):
            self.fault_p[rid] = g[[f"p_{c}" for c in CLASSES]].to_numpy()
            if all(c in g.columns for c in kcols):
                self.knock[rid] = g[kcols].to_numpy()
        self.physics_rules = load_config()["physics"]
        self.sev_p: dict[str, np.ndarray] = {}
        sev_cols = [f"p_{s:.3f}" for s in SIZES]
        for rid, g in sev_oof.sort_values(["recording_id", "window_idx"]).groupby("recording_id"):
            self.sev_p[rid] = g[sev_cols].to_numpy()

    @staticmethod
    def recording_id(fault: str | None, size: float, load: int) -> str:
        if fault is None:
            return f"Normal_{load}"
        if fault == "Outer race" and size > 0.021:  # CWRU has no 0.028" outer race recording
            size = 0.021
        prefix = {"Inner race": "IR", "Ball": "B", "Outer race": "OR"}[fault]
        pos = "@6" if fault == "Outer race" else ""
        return f"{prefix}{round(size * 1000):03d}{pos}_{load}"

    def _chunk(self, arr: np.ndarray, k: int, rng: np.random.Generator, start: int | None = None):
        k = min(k, len(arr))
        if start is None:
            start = int(rng.integers(0, len(arr) - k + 1))
        return arr[start:start + k], start

    def sample(self, fault: str | None, size: float, load: int, k: int, rng: np.random.Generator) -> tuple:
        """Same decision as the real app: ML vote, overridden by a clear knock signature."""
        from pdm.physics import knock_evidence

        rid = self.recording_id(fault, size, load)
        chunk, start = self._chunk(self.fault_p[rid], k, rng)
        p = chunk.mean(axis=0)
        j = int(p.argmax())
        diag, conf = CLASSES[j], float(p[j])
        if rid in self.knock:
            kn, _ = self._chunk(self.knock[rid], k, rng, start)
            med = dict(zip(["e_bpfo_1", "e_bpfi_1", "e_bsf_1", "e_bsf_2"], np.median(kn, axis=0)))
            call = knock_evidence(med, self.physics_rules).call
            if call and call != diag:
                diag, conf = call, float(p[CLASSES.index(call)])
        est_size = None
        if diag != "Normal":
            sev_rid = rid if rid in self.sev_p else self.recording_id("Inner race", 0.007, load)
            sp = self._chunk(self.sev_p[sev_rid], k, rng)[0].mean(axis=0)
            est_size = SIZES[int(sp.argmax())]
        return diag, conf, est_size


@dataclass
class Measurement:
    diag: str = "Normal"
    conf: float = 1.0
    est_size: float | None = None
    health: str = "Healthy"
    low_conf: bool = False


def encode_observation(m: Measurement, load: int, monitoring_on: bool, days_since_measurement: int,
                       days_since_inspection: int, inspected_degradation: float, inspection_found_fault: bool
                       ) -> np.ndarray:
    o = np.zeros(OBS_DIM, dtype=np.float32)
    o[CLASSES.index(m.diag)] = 1.0
    o[4] = m.conf
    o[5] = 0.0 if m.est_size is None else m.est_size / 0.028
    o[6 + HEALTH_LEVELS.index(m.health)] = 1.0
    o[9] = float(m.low_conf)
    o[10] = load / 3.0
    o[11] = float(monitoring_on)
    o[12] = min(days_since_measurement, 14) / 14.0
    o[13] = min(days_since_inspection, 180) / 180.0
    o[14] = inspected_degradation
    o[15] = float(inspection_found_fault)
    return o


# ------------------------------------------------------------------ environment
class BearingMaintenanceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: dict | None = None, obs_model: ObservationModel | None = None, fixed_load: int | None = None):
        super().__init__()
        self.cfg = cfg or load_config()["rl"]
        self.health_rules = load_config()["health"]
        self.obs_model = obs_model or ObservationModel()
        self.fixed_load = fixed_load
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)

    # ---- hidden-state helpers
    def _true_size(self) -> float:
        th = self.cfg["degradation"]["size_thresholds"]
        return SIZES[int(np.searchsorted(th, self.degradation, side="right"))]

    def _measure(self) -> None:
        o = self.cfg["observation"]
        k = o["monitoring_windows"] if self.monitoring_days > 0 else o["windows_per_measurement"]
        load = 0 if self.reduced_today else self.load
        diag, conf, size = self.obs_model.sample(self.fault if self.initiated else None, self._true_size(),
                                                 load, k, self.np_random)
        h = assess_health(diag, conf, size, self.load, self.health_rules)
        self.last = Measurement(diag, conf, size, h.status, h.low_confidence)
        self.days_since_measurement = 0

    def _obs(self) -> np.ndarray:
        return encode_observation(self.last, self.load, self.monitoring_days > 0, self.days_since_measurement,
                                  self.days_since_inspection, self.inspected_degradation, self.inspection_found_fault)

    def _info(self) -> dict:
        return {"day": self.day, "true_fault": self.fault if self.initiated else "Normal",
                "degradation": float(self.degradation), "true_size": self._true_size() if self.initiated else None,
                "load": self.load, "bearing_age_days": self.bearing_age, "health_observed": self.last.health,
                "diagnosis": self.last.diag, "confidence": self.last.conf, "est_size": self.last.est_size,
                "failures": self.n_failures, "maintenances": self.n_maint, "unnecessary_maintenances": self.n_unnec,
                "inspections": self.n_insp, "downtime_days": self.downtime, "producing_days": self.producing_days,
                "safety_days": self.safety_days}

    def _new_bearing(self) -> None:
        self.fault = FAULTS[self.np_random.integers(len(FAULTS))]  # the type it WILL develop, if any
        self.initiated = False
        self.degradation = 0.0
        self.bearing_age = 0
        self.inspected_degradation = 0.0
        self.inspection_found_fault = False
        self.days_since_inspection = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        d = self.cfg["degradation"]
        self.load = int(self.fixed_load if self.fixed_load is not None else self.np_random.integers(0, 4))
        self._new_bearing()
        self.days_since_inspection = 60  # nobody has looked recently
        if self.np_random.random() < d["initial_fault_prob"]:
            self.initiated = True
            self.degradation = float(self.np_random.uniform(0, d["initial_fault_max"]))
        self.day = 0
        self.monitoring_days = 0
        self.reduced_today = False
        self.n_failures = self.n_maint = self.n_unnec = self.n_insp = 0
        self.downtime = self.producing_days = self.safety_days = 0
        self._measure()
        return self._obs(), self._info()

    def _wear_one_day(self, reduced: bool) -> bool:
        """Advance hidden degradation by one operating day. Returns True if the bearing fails."""
        d = self.cfg["degradation"]
        lf = d["load_factor"][self.load] * (d["reduced_load_factor"] if reduced else 1.0)
        if not self.initiated:
            if self.np_random.random() < d["initiation_prob_per_day"]:
                self.initiated = True
            return False
        mean = lf * (1.0 + self.degradation) / d["growth_days_to_failure"]
        shape = d["growth_gamma_shape"]
        self.degradation = float(min(1.0, self.degradation + self.np_random.gamma(shape, mean / shape)))
        p_fail = min(0.9, d["failure_a"] * self.degradation ** d["failure_b"] * lf)
        return bool(self.np_random.random() < p_fail)

    def step(self, action: int):
        c, o = self.cfg["costs"], self.cfg["observation"]
        a = int(action)
        reward, days, terminated = 0.0, 1, False
        self.reduced_today = a == 3

        if a in (0, 1, 3):  # the machine runs today
            if a == 1:
                if self.monitoring_days == 0:
                    reward -= c["monitoring_activation"]
                self.monitoring_days = o["monitoring_duration_days"]
            reward += c["production_per_day"] * (c["reduced_load_fraction"] if a == 3 else 1.0)
            if self.monitoring_days > 0:
                reward -= c["monitoring_per_day"]
            if self.initiated and self.degradation >= self.cfg["degradation"]["critical_degradation"]:
                reward -= c["safety_per_day"]
                self.safety_days += 1
            self.producing_days += 1
            if self._wear_one_day(reduced=a == 3):
                reward -= c["failure"]
                self.n_failures += 1
                terminated = True
        elif a == 2:  # inspection: downtime, reveals the truth
            reward -= c["inspection"]
            days = c["inspection_downtime_days"]
            self.n_insp += 1
            self.inspected_degradation = float(self.degradation) if self.initiated else 0.0
            self.inspection_found_fault = bool(self.initiated)
            self.days_since_inspection = -days  # becomes 0 after the day counter below
        elif a == 4:  # replace the bearing
            reward -= c["maintenance"]
            if not self.initiated:
                reward -= c["unnecessary_maintenance"]
                self.n_unnec += 1
            days = c["maintenance_downtime_days"]
            self.n_maint += 1
            self._new_bearing()
            self.days_since_inspection = -days
            self.inspection_found_fault = False
            self.inspected_degradation = 0.0

        if a in (2, 4):
            self.downtime += days
        self.day += days
        self.bearing_age += days
        self.days_since_inspection += days
        self.days_since_measurement += days
        if a != 1 and self.monitoring_days > 0:
            self.monitoring_days = max(0, self.monitoring_days - days)
        elif a == 1:
            self.monitoring_days -= 1

        truncated = (not terminated) and self.day >= self.cfg["episode_days"]
        if not terminated:
            interval = 1 if self.monitoring_days > 0 else o["measure_every_days"]
            if a == 4 or self.days_since_measurement >= interval:
                self._measure()
        return self._obs(), float(reward), terminated, truncated, self._info()


def make_env(seed: int = 0, **kw) -> BearingMaintenanceEnv:
    env = BearingMaintenanceEnv(**kw)
    env.reset(seed=seed)
    return env

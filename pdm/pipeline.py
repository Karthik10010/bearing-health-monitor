"""analyze(signal, fs, rpm, load) -> AnalysisReport.

This is the ONE place where a recording is turned into results. The dashboard and the chat assistant
only display / explain the AnalysisReport; they never compute or invent results themselves.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np

from pdm.config import load_config, path
from pdm.data.loader import resample_to
from pdm.health import assess_health
from pdm.physics import knock_evidence
from pdm.signal.bearing import defect_frequencies
from pdm.signal.features import envelope_spectrum, extract_features, feature_names, params_from_config
from pdm.signal.windowing import window_signal

CAVEATS = [
    "Models were trained and tested only on the CWRU laboratory test rig (2 HP motor, 6205 bearing), "
    "not on a real PA fan. The PA fan is the target application, not a validated deployment.",
    "CWRU faults are artificially machined pits, not naturally grown damage.",
    "On bearings the model has never seen, fault-TYPE accuracy is much lower than on seen bearings; a physics "
    "knock-signature check improves it (see model_evaluation). Normal-vs-faulty detection holds up much better. "
    "Fault-SIZE estimates did not transfer to unseen fault types and are indicative only.",
    "Health status is a RULE-BASED ESTIMATE from the predicted class, confidence and estimated size.",
    "The maintenance recommendation comes from an RL agent trained in a SIMULATION with illustrative "
    "(assumed) costs and wear rates - it shows the method, not a validated plant policy.",
    "Edge-AI timings were measured on a development laptop, not on edge hardware.",
]


@dataclass
class AnalysisReport:
    created_utc: str
    input: dict
    defect_frequencies_hz: dict
    envelope_peaks: dict
    diagnosis: dict
    health: dict
    recommendation: dict | None
    whatif_continue: dict | None
    top_features: list
    model_evaluation: dict
    caveats: list
    warnings: list
    arrays: dict = field(default_factory=dict, repr=False)   # plot data; NOT sent to the LLM

    def to_dict(self, include_arrays: bool = False) -> dict:
        d = asdict(self)
        if not include_arrays:
            d.pop("arrays")
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=float)


FINGERPRINT_N = 256


def signal_fingerprint(x: np.ndarray, n: int = FINGERPRINT_N) -> list[float]:
    """First n raw samples - identifies a CWRU training recording (also after float32 storage)."""
    return [round(float(v), 6) for v in np.asarray(x, dtype=np.float64).ravel()[:n]]


@lru_cache(maxsize=1)
def _training_fingerprints() -> dict[str, np.ndarray]:
    p = path("models_dir") / "training_fingerprints.json"
    if not p.exists():
        return {}
    return {k: np.asarray(v) for k, v in json.loads(p.read_text(encoding="utf-8")).items()}


def matches_training_recording(x: np.ndarray) -> str | None:
    head = np.asarray(x, dtype=np.float64).ravel()[:FINGERPRINT_N]
    if head.size < FINGERPRINT_N:
        return None
    for rid, ref in _training_fingerprints().items():
        if np.allclose(head, ref, atol=1e-5, rtol=0):
            return rid
    return None


# ------------------------------------------------------------------ cached resources (loaded once)
@lru_cache(maxsize=1)
def _predictor():
    from pdm.ml.predict import Predictor
    return Predictor()


@lru_cache(maxsize=1)
def _agent():
    if not (path("models_dir") / "ppo_agent.zip").exists():
        return None
    from pdm.rl.agent import Agent
    return Agent()


@lru_cache(maxsize=1)
def _obs_model():
    from pdm.rl.env import ObservationModel
    return ObservationModel()


@lru_cache(maxsize=1)
def _reference() -> dict:
    m = path("metrics_dir")
    stats = json.loads((m / "class_feature_stats.json").read_text(encoding="utf-8"))
    fault = json.loads((m / "fault_metrics.json").read_text(encoding="utf-8"))
    sev = json.loads((m / "severity_metrics.json").read_text(encoding="utf-8"))
    ev = {
        "fault_type_seen_bearings_lolo": {
            "window_macro_f1": round(fault["window"]["macro_f1"]["mean"], 3),
            "recording_accuracy": round(fault["recording"]["accuracy"]["mean"], 3)},
        "fault_type_unseen_bearings": {
            "window_macro_f1": round(fault["leave_one_size_out"]["window"]["macro_f1"]["mean"], 3),
            "recording_accuracy": round(fault["leave_one_size_out"]["recording"]["accuracy"]["mean"], 3),
            "normal_windows_correct": fault["leave_one_size_out"]["window_confusion_normalised"][0][0]},
        "fault_size_seen_bearings_lolo": {"window_macro_f1": round(sev["window"]["macro_f1"]["mean"], 3)},
        "fault_size_unseen_fault_type": {
            "window_macro_f1": round(sev["leave_one_fault_type_out"]["window"]["macro_f1"]["mean"], 3)},
        "naive_random_split_window_macro_f1": round(fault["naive_random_split"]["macro_f1"], 3),
    }
    if "hybrid_unseen_bearings" in fault:
        hy = fault["hybrid_unseen_bearings"]
        ev["hybrid_unseen_bearings"] = {"ml_only_accuracy": hy["ml_only_accuracy"],
                                        "hybrid_accuracy": hy["hybrid_accuracy"],
                                        "physics_calls": hy["physics_calls"],
                                        "physics_calls_correct": hy["physics_calls_correct"]}
    rl_file = m / "rl_evaluation.json"
    if rl_file.exists():
        rl = json.loads(rl_file.read_text(encoding="utf-8"))
        ev["rl_simulated_test"] = {k: {"mean_reward": round(v["mean_reward"], 1),
                                       "failures_per_episode": round(v["failures_per_episode"], 3),
                                       "availability_pct": round(v["availability_pct"], 1)}
                                   for k, v in rl["summary"].items() if not k.startswith("PPO seed")}
    return {"stats": stats, "importance": fault["feature_importance"], "evaluation": ev}


def _top_features(feat_median: dict, predicted: str, n: int = 8) -> list:
    ref = _reference()
    healthy = ref["stats"]["Normal"]
    out = []
    for name in list(ref["importance"])[:n]:
        v, h = feat_median[name], healthy[name]["median"]
        out.append({"feature": name, "value": round(v, 4), "healthy_median": round(h, 4),
                    "healthy_p25_p75": [round(healthy[name]["p25"], 4), round(healthy[name]["p75"], 4)],
                    "predicted_class_median": round(ref["stats"][predicted][name]["median"], 4),
                    "ratio_to_healthy": round(v / h, 2) if abs(h) > 1e-9 else None,
                    "importance": round(ref["importance"][name], 4)})
    return out


# ------------------------------------------------------------------ main entry point
def analyze(signal: np.ndarray, fs: float, rpm: float, load_hp: float, source: str = "upload",
            run_whatif: bool = True, whatif_rollouts: int = 300) -> AnalysisReport:
    cfg = load_config()
    target_fs = cfg["data"]["target_fs"]
    params = params_from_config()
    warnings: list[str] = []

    x = np.asarray(signal, dtype=np.float64).ravel()
    if x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("the signal is empty or contains NaN/inf values")
    if np.std(x) < 1e-12:
        raise ValueError("the signal is flat (constant) - is the sensor connected?")
    if fs <= 0 or rpm <= 0:
        raise ValueError("sampling rate and RPM must be positive")
    if fs < target_fs:
        warnings.append(f"sampling rate {fs:g} Hz is below the model's {target_fs} Hz; high-frequency "
                        "information is missing, so results may be unreliable")
    if not 1650 <= rpm <= 1850:
        warnings.append(f"RPM {rpm:g} is outside the CWRU training range (~1720-1800 RPM)")
    if not 0 <= load_hp <= 3:
        warnings.append(f"load {load_hp:g} HP is outside the CWRU training range (0-3 HP)")
    seen = matches_training_recording(x)
    if seen:
        warnings.append(f"this signal is CWRU recording {seen}, which the models were TRAINED on - the result "
                        "is not an independent test (expect very high confidence)")

    x12 = resample_to(x, fs, target_fs)
    win = window_signal(x12)
    if len(win) < 6:
        warnings.append(f"short recording ({x12.size / target_fs:.2f} s, {len(win)} windows); "
                        "at least 1 s is recommended")
    X = extract_features(win, target_fs, rpm, params)
    names = feature_names(params)
    feat_median = {n: float(v) for n, v in zip(names, np.median(X, axis=0))}

    # ---- two opinions: the ML model, and the physics knock-signature check
    predictor = _predictor()
    pred = predictor.predict(X, names)
    ev = knock_evidence(feat_median)
    final = ev.call or pred.fault
    agree = ev.call is None or ev.call == pred.fault
    confidence = float(pred.probabilities.get(final, 0.0))  # ML vote share for the final class
    size, size_p, size_conf = pred.size_in, pred.size_probabilities, pred.size_confidence
    if final != "Normal" and size is None:  # physics found a fault the ML model called Normal
        size, size_p, size_conf = predictor.severity_for(X, names)
    if final == "Normal":
        size = size_p = size_conf = None
    note = None if agree else (f"the ML model says {pred.fault} ({pred.confidence:.0%}) but the vibration shows a "
                               f"clear {ev.call.lower()} knock signature, which was used")
    health = assess_health(final, confidence, size, load_hp, note=note)

    dfreq = defect_frequencies(rpm)
    env_peaks = {}
    for d in ("bpfo", "bpfi", "bsf"):
        env_peaks[d] = {"frequency_hz": round(getattr(dfreq, d), 2),
                        **{f"peak_ratio_h{h}": round(feat_median[f"e_{d}_{h}"], 2) for h in (1, 2, 3)},
                        "healthy_median_h1": round(_reference()["stats"]["Normal"][f"e_{d}_1"]["median"], 2)}
    env_peaks["shaft_1x"] = {"frequency_hz": round(dfreq.shaft_hz, 2), "peak_ratio_h1": round(feat_median["e_shaft_1"], 2)}
    env_peaks["meaning"] = ("peak height near each defect frequency in the envelope spectrum divided by the "
                            "spectrum's background level; ~1-2 = nothing special, >5 = a clear repeating knock")

    rec = None
    agent = _agent()
    if agent is not None:
        from pdm.rl.agent import build_observation
        obs, assumptions = build_observation(final, confidence, size, health.status,
                                             health.low_confidence, load_hp)
        rec = agent.recommend(obs, assumptions).to_dict()
    else:
        warnings.append("RL agent not trained yet (models/ppo_agent.zip missing) - no recommendation")

    whatif = None
    if run_whatif:
        from pdm.rl.whatif import simulate_continue
        p_sim = pred.probabilities if agree else {c: float(c == final) for c in pred.probabilities}
        whatif = simulate_continue(p_sim, size, int(min(3, max(0, round(load_hp)))),
                                   n_rollouts=whatif_rollouts, obs_model=_obs_model())

    # ---- plot data (kept out of the LLM context)
    n_fft = min(x12.size, 1 << 16)
    seg = x12[:n_fft] - x12[:n_fft].mean()
    spec = np.abs(np.fft.rfft(seg * np.hanning(n_fft))) / n_fft
    f_spec = np.fft.rfftfreq(n_fft, 1 / target_fs)
    ef, em = envelope_spectrum(win, target_fs, params)
    keep = ef <= params["envelope_max_hz"]
    step = max(1, len(f_spec) // 4000)
    arrays = {"signal_12k": x12.astype(np.float32), "fs_used": target_fs,
              "fft_freq": f_spec[::step], "fft_mag": spec[::step],
              "env_freq": ef[keep], "env_mag": em[:, keep].mean(axis=0),
              "window_predictions": pred.window_predictions, "features_per_window": X.astype(np.float32),
              "feature_names": names,
              "feature_table": [{"feature": n, "group": {"t": "time", "f": "frequency", "e": "envelope"}[n[0]],
                                 "value (median over windows)": round(feat_median[n], 4),
                                 "healthy median": round(_reference()["stats"]["Normal"][n]["median"], 4),
                                 f"{final} median": round(_reference()["stats"][final][n]["median"], 4)}
                                for n in names]}

    diag = {"fault": final, "confidence": confidence,
            "decision_source": "knock signature (physics)" if ev.call else "ML model",
            "methods_agree": agree,
            "ml_fault": pred.fault, "ml_confidence": pred.confidence, "probabilities": pred.probabilities,
            "window_agreement": float(np.mean([w == final for w in pred.window_predictions])),
            "knock_evidence": ev.to_dict(),
            "size_in": size, "size_probabilities": size_p, "size_confidence": size_conf, "n_windows": len(win)}
    return AnalysisReport(
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        input={"source": source, "fs_original_hz": float(fs), "fs_used_hz": target_fs, "rpm": float(rpm),
               "load_hp": float(load_hp), "duration_s": round(x.size / fs, 3), "samples": int(x.size),
               "window_size": cfg["signal"]["window_size"], "bearing": cfg["bearing"]["name"]},
        defect_frequencies_hz={k: round(v, 2) for k, v in dfreq.as_dict().items()},
        envelope_peaks=env_peaks, diagnosis=diag, health=health.to_dict(), recommendation=rec,
        whatif_continue=whatif, top_features=_top_features(feat_median, final),
        model_evaluation=_reference()["evaluation"], caveats=list(CAVEATS), warnings=warnings, arrays=arrays,
    )

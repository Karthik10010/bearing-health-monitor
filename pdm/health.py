"""Rule-based health status (Healthy / Warning / Critical).

This is a RULE-BASED ESTIMATE, not a measured condition. Thresholds live in config.yaml (health:).

| Condition                                          | Health                            |
|----------------------------------------------------|-----------------------------------|
| Predicted Normal with confidence >= 0.70           | Healthy                           |
| Fault predicted, estimated size 0.007"             | Warning (early stage)             |
| Fault predicted, estimated size 0.014"             | Warning (moderate); Critical at load >= 2 HP |
| Fault predicted, estimated size >= 0.021"          | Critical                          |
| Top-class confidence < 0.60 (any class)            | + "Low confidence - re-measure" flag |
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from pdm.config import load_config

HEALTH_LEVELS = ["Healthy", "Warning", "Critical"]
LABEL = "Rule-based estimate"


@dataclass(frozen=True)
class HealthStatus:
    status: str          # Healthy | Warning | Critical
    stage: str           # short human description
    low_confidence: bool
    reasons: list[str]
    condition_index: int = 0   # 0-100 needle position for the dashboard gauge (see CONDITION_INDEX)
    label: str = LABEL

    def to_dict(self) -> dict:
        return asdict(self)


# Where the gauge needle points. It is NOT a measured quantity: a fixed position per rule outcome
# (Healthy 0-33, Warning 33-66, Critical 66-100), so the needle can never contradict the status.
CONDITION_INDEX = {"healthy": 15, "uncertain_normal": 40, "early": 47, "moderate": 58,
                   "moderate_high_load": 72, "advanced": 84, "advanced_largest": 94}


def assess_health(fault: str, confidence: float, size_in: float | None, load_hp: float,
                  rules: dict | None = None, note: str | None = None) -> HealthStatus:
    """note: extra reason to show (e.g. the ML model and the knock signature disagree); it also
    raises the low-confidence flag, because two independent methods disagree."""
    r = rules or load_config()["health"]
    reasons: list[str] = []
    low = confidence < r["low_confidence_threshold"] or note is not None
    if confidence < r["low_confidence_threshold"]:
        reasons.append(f"top-class confidence {confidence:.2f} < {r['low_confidence_threshold']:.2f}: re-measure")
    if note:
        reasons.append(note + " - an inspection is advisable")

    if fault == "Normal":
        if confidence >= r["healthy_min_confidence"]:
            status, stage, ci = "Healthy", "no fault detected", CONDITION_INDEX["healthy"]
            reasons.insert(0, f"predicted Normal with confidence {confidence:.2f} >= {r['healthy_min_confidence']:.2f}")
        else:
            # Normal, but not confidently: do not claim Healthy.
            status, stage, ci = "Warning", "uncertain - possibly early fault", CONDITION_INDEX["uncertain_normal"]
            reasons.insert(0, f"predicted Normal but confidence {confidence:.2f} < {r['healthy_min_confidence']:.2f}")
        return HealthStatus(status, stage, low, reasons, ci)

    if size_in is None:
        raise ValueError("a fault was predicted but no size estimate was given")
    s = round(float(size_in), 3)
    if s >= r["critical_min_size"]:
        status, stage = "Critical", "advanced damage"
        ci = CONDITION_INDEX["advanced_largest" if s >= 0.028 else "advanced"]
        reasons.insert(0, f'{fault} with estimated size {s:.3f}" >= {r["critical_min_size"]:.3f}"')
    elif abs(s - r["escalate_size"]) < 1e-9 and load_hp >= r["escalate_min_load_hp"]:
        status, stage, ci = "Critical", "moderate damage under high load", CONDITION_INDEX["moderate_high_load"]
        reasons.insert(0, f'{fault} {s:.3f}" at load {load_hp:g} HP >= {r["escalate_min_load_hp"]} HP')
    elif abs(s - r["escalate_size"]) < 1e-9:
        status, stage, ci = "Warning", "moderate damage", CONDITION_INDEX["moderate"]
        reasons.insert(0, f'{fault} with estimated size {s:.3f}"')
    else:
        status, stage, ci = "Warning", "early-stage damage", CONDITION_INDEX["early"]
        reasons.insert(0, f'{fault} with estimated size {s:.3f}"')
    return HealthStatus(status, stage, low, reasons, ci)

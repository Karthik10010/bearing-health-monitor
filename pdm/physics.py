"""Physics cross-check: is there a CLEAR knock signature at one fault's defect frequency?

No training involved. Each fault knocks at its own rate (BPFO / BPFI / BSF x shaft speed). If one
family's envelope peak is both strong (>= min_peak_ratio x the background) and clearly the biggest
(>= min_dominance x the next family), that is direct physical evidence of WHERE the damage is.

On the 48 CWRU recordings this rule made a call on 19, all correct; it stays silent on the rest
(healthy bearings and damaged bearings without a clear signature), and then the ML model decides.
The thresholds live in config.yaml (physics:).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from pdm.config import load_config

FAMILIES = {"Outer race": "bpfo", "Inner race": "bpfi", "Ball": "bsf"}


@dataclass
class KnockEvidence:
    call: str | None                 # fault class, or None = no clear signature
    peak_ratios: dict[str, float]    # strongest peak per family (x background), median over windows
    strongest: str
    dominance: float                 # strongest / second strongest
    rule: str

    def to_dict(self) -> dict:
        return asdict(self)


def family_peaks(feat: dict[str, float]) -> dict[str, float]:
    """Strongest 1x peak per family; ball faults often show at 2x BSF, so BSF uses max(1x, 2x)."""
    return {"Outer race": float(feat["e_bpfo_1"]), "Inner race": float(feat["e_bpfi_1"]),
            "Ball": float(max(feat["e_bsf_1"], feat["e_bsf_2"]))}


def knock_evidence(feat_median: dict[str, float], rules: dict | None = None) -> KnockEvidence:
    r = rules or load_config()["physics"]
    peaks = family_peaks(feat_median)
    order = sorted(peaks, key=peaks.get, reverse=True)
    top, second = peaks[order[0]], peaks[order[1]]
    dom = top / second if second > 0 else float("inf")
    clear = top >= r["min_peak_ratio"] and dom >= r["min_dominance"]
    rule = (f"strongest knock family must be >= {r['min_peak_ratio']:g}x background and "
            f">= {r['min_dominance']:g}x the next family")
    return KnockEvidence(order[0] if clear else None, {k: round(v, 2) for k, v in peaks.items()}, order[0],
                         round(float(dom), 2), rule)


def knock_call_from_matrix(X: np.ndarray, names: list[str], rules: dict | None = None) -> str | None:
    """Same rule on a (windows x features) matrix, using the median over windows."""
    idx = {n: i for i, n in enumerate(names)}
    med = {k: float(np.median(X[:, idx[k]])) for k in ("e_bpfo_1", "e_bpfi_1", "e_bsf_1", "e_bsf_2")}
    return knock_evidence(med, rules).call

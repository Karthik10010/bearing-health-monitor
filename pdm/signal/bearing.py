"""6205 drive-end bearing defect frequencies.

Multipliers are "times shaft speed" from the CWRU bearing-information page and live in config.yaml.
Example: at 1750 RPM the shaft turns 29.17 times per second, so the outer-race defect
frequency is 29.17 x 3.5848 = 104.6 Hz (104.6 "knocks" per second).
"""
from __future__ import annotations

from dataclasses import dataclass

from pdm.config import load_config


@dataclass(frozen=True)
class DefectFrequencies:
    shaft_hz: float
    bpfi: float
    bpfo: float
    bsf: float
    ftf: float

    def as_dict(self) -> dict[str, float]:
        return {"shaft": self.shaft_hz, "bpfi": self.bpfi, "bpfo": self.bpfo,
                "bsf": self.bsf, "ftf": self.ftf}


def defect_frequencies(rpm: float) -> DefectFrequencies:
    if rpm <= 0:
        raise ValueError(f"rpm must be positive, got {rpm}")
    b = load_config()["bearing"]
    fr = rpm / 60.0
    return DefectFrequencies(
        shaft_hz=fr,
        bpfi=b["bpfi"] * fr,
        bpfo=b["bpfo"] * fr,
        bsf=b["bsf"] * fr,
        ftf=b["ftf"] * fr,
    )

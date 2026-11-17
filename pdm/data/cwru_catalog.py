"""Catalogue of the CWRU recordings used in this project.

Source (checked 2026-11-13):
  https://engineering.case.edu/bearingdatacenter/normal-baseline-data
  https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data

Scope: drive-end 6205 bearing; Normal + Inner race + Ball + Outer race (@6:00, load-zone centred);
all four loads. Fault data are the 12 kHz drive-end files; normal baseline files are 48 kHz.
0.028" faults exist only for inner race and ball (no outer race), as on the CWRU site.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

BASE_URL = "https://engineering.case.edu/sites/default/files/{num}.mat"

CLASSES = ["Normal", "Inner race", "Outer race", "Ball"]
FAULT_SIZES = [0.007, 0.014, 0.021, 0.028]       # inches
LOAD_RPM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}  # approx. motor speed per load (HP)


@dataclass(frozen=True)
class CwruFile:
    file_number: int
    name: str          # CWRU label, e.g. "IR007_0"
    fault: str         # one of CLASSES
    size_in: float     # fault diameter in inches; 0.0 for Normal
    load_hp: int
    rpm: int           # approximate RPM from the CWRU table
    fs: int            # original sampling rate (Hz)

    @property
    def url(self) -> str:
        return BASE_URL.format(num=self.file_number)

    @property
    def filename(self) -> str:
        return f"{self.file_number}.mat"

    @property
    def recording_id(self) -> str:
        return self.name

    def to_dict(self) -> dict:
        d = asdict(self)
        d["url"] = self.url
        return d


# (size, load) -> file number, copied from the official tables.
_NORMAL = {0: 97, 1: 98, 2: 99, 3: 100}
_INNER = {
    (0.007, 0): 105, (0.007, 1): 106, (0.007, 2): 107, (0.007, 3): 108,
    (0.014, 0): 169, (0.014, 1): 170, (0.014, 2): 171, (0.014, 3): 172,
    (0.021, 0): 209, (0.021, 1): 210, (0.021, 2): 211, (0.021, 3): 212,
    (0.028, 0): 3001, (0.028, 1): 3002, (0.028, 2): 3003, (0.028, 3): 3004,
}
_BALL = {
    (0.007, 0): 118, (0.007, 1): 119, (0.007, 2): 120, (0.007, 3): 121,
    (0.014, 0): 185, (0.014, 1): 186, (0.014, 2): 187, (0.014, 3): 188,
    (0.021, 0): 222, (0.021, 1): 223, (0.021, 2): 224, (0.021, 3): 225,
    (0.028, 0): 3005, (0.028, 1): 3006, (0.028, 2): 3007, (0.028, 3): 3008,
}
_OUTER_AT6 = {
    (0.007, 0): 130, (0.007, 1): 131, (0.007, 2): 132, (0.007, 3): 133,
    (0.014, 0): 197, (0.014, 1): 198, (0.014, 2): 199, (0.014, 3): 200,
    (0.021, 0): 234, (0.021, 1): 235, (0.021, 2): 236, (0.021, 3): 237,
}
_PREFIX = {"Inner race": "IR", "Ball": "B", "Outer race": "OR"}


def _build() -> list[CwruFile]:
    out: list[CwruFile] = []
    for load, num in _NORMAL.items():
        out.append(CwruFile(num, f"Normal_{load}", "Normal", 0.0, load, LOAD_RPM[load], 48000))
    for fault, table in (("Inner race", _INNER), ("Ball", _BALL), ("Outer race", _OUTER_AT6)):
        for (size, load), num in table.items():
            mils = f"{round(size * 1000):03d}"
            pos = "@6" if fault == "Outer race" else ""
            name = f"{_PREFIX[fault]}{mils}{pos}_{load}"
            out.append(CwruFile(num, name, fault, size, load, LOAD_RPM[load], 12000))
    return out


CATALOG: list[CwruFile] = _build()

# Independent hold-out: outer-race faults at 3 and 12 o'clock (same rig and bearing type, but different
# physical bearings). NEVER used for training or design decisions - only as a final check and as genuine
# "never seen" recordings to upload in the dashboard. Downloaded into data/unseen/.
HOLDOUT: list[CwruFile] = [
    CwruFile(144, "OR007@3_0", "Outer race", 0.007, 0, LOAD_RPM[0], 12000),
    CwruFile(156, "OR007@12_0", "Outer race", 0.007, 0, LOAD_RPM[0], 12000),
    CwruFile(246, "OR021@3_0", "Outer race", 0.021, 0, LOAD_RPM[0], 12000),
    CwruFile(258, "OR021@12_0", "Outer race", 0.021, 0, LOAD_RPM[0], 12000),
]


def by_number(num: int) -> CwruFile:
    for f in CATALOG:
        if f.file_number == num:
            return f
    raise KeyError(f"File number {num} is not in the catalog")

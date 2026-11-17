"""Small demo recordings for the dashboard (data/samples/), cut from the CWRU files.

IMPORTANT: the final models are trained on ALL CWRU recordings, so these samples are NOT an independent
test - the dashboard says so (analyze() detects them by fingerprint).
"""
from __future__ import annotations

import json

import numpy as np

from pdm.config import path
from pdm.data.cwru_catalog import by_number
from pdm.data.loader import read_cwru_mat

# (file number, seconds, file format)
SAMPLES = [(98, 3.0, "npy"), (105, 3.0, "npy"), (199, 3.0, "csv"), (224, 3.0, "npy"), (235, 3.0, "npy"),
           (3003, 3.0, "npy")]


def make_samples() -> list[dict]:
    out_dir = path("samples_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for num, seconds, fmt in SAMPLES:
        e = by_number(num)
        ms = read_cwru_mat(path("raw_dir") / e.filename, e.file_number)
        x = ms.signal[: int(seconds * e.fs)].astype(np.float32)
        name = f"cwru_{e.recording_id.replace('@', 'at')}.{fmt}"
        if fmt == "npy":
            np.save(out_dir / name, x)
        else:
            np.savetxt(out_dir / name, x, fmt="%.6f", header="signal", comments="")
        index.append({"file": name, "label": f"{e.fault}" + (f' {e.size_in:.3f}"' if e.size_in else "")
                      + f" @ {e.load_hp} HP", "fs": e.fs, "rpm": float(ms.rpm or e.rpm), "load_hp": e.load_hp,
                      "true_fault": e.fault, "true_size_in": e.size_in or None, "cwru_recording": e.recording_id,
                      "note": "Cut from a CWRU recording used in training - not an independent test."})
    (out_dir / "samples.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index

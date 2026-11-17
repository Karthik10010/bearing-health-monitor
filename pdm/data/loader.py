"""Read vibration signals (CWRU .mat files and user uploads) and resample them to 12 kHz.

Files are treated strictly as data: .mat via scipy.io.loadmat, .npy via np.load(allow_pickle=False),
.csv via pandas. Nothing inside a file is ever executed.
"""
from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import resample_poly

from pdm.config import load_config, path

log = logging.getLogger(__name__)

_DE_KEY = re.compile(r"^X(\d+)_DE_time$")
_RPM_KEY = re.compile(r"^X(\d+)RPM$")


@dataclass
class MatSignal:
    signal: np.ndarray
    key: str
    rpm: float | None
    notes: list[str] = field(default_factory=list)


def read_cwru_mat(source: str | Path | BinaryIO, file_number: int | None = None) -> MatSignal:
    """Extract the drive-end signal from a CWRU-style .mat file.

    CWRU quirk: some files contain more than one ``X???_DE_time`` variable (e.g. 99.mat also holds
    X098_DE_time). We take the key whose number matches the file number. If no number is given or
    there is no match, a file with exactly one DE key is accepted (with a note); otherwise we fail
    loudly instead of guessing.
    """
    mat = scipy.io.loadmat(source)
    de_keys = {k: int(m.group(1)) for k in mat if (m := _DE_KEY.match(k))}
    rpm_keys = {int(m.group(1)): k for k in mat if (m := _RPM_KEY.match(k))}
    notes: list[str] = []
    if not de_keys:
        raise ValueError("no '*_DE_time' (drive-end) variable found in .mat file")

    chosen: str | None = None
    if file_number is not None:
        chosen = next((k for k, n in de_keys.items() if n == file_number), None)
        if chosen is None:
            notes.append(f"no DE key for file {file_number}; keys present: {sorted(de_keys)}")
    if chosen is None:
        if len(de_keys) != 1:
            raise ValueError(f"ambiguous drive-end variables {sorted(de_keys)}; cannot choose safely")
        chosen = next(iter(de_keys))
    if len(de_keys) > 1:
        notes.append(f"multiple DE keys {sorted(de_keys)}; used {chosen}")

    sig = np.asarray(mat[chosen], dtype=np.float64).ravel()
    num = de_keys[chosen]
    rpm = float(np.asarray(mat[rpm_keys[num]]).ravel()[0]) if num in rpm_keys else None
    if rpm is None and len(rpm_keys) == 1:
        rpm = float(np.asarray(mat[next(iter(rpm_keys.values()))]).ravel()[0])
        notes.append("RPM taken from the only RPM variable in the file")
    for n in notes:
        log.info("%s: %s", file_number, n)
    return MatSignal(signal=sig, key=chosen, rpm=rpm, notes=notes)


def resample_to(signal: np.ndarray, fs: float, target_fs: float | None = None) -> np.ndarray:
    """Resample with an anti-aliasing polyphase filter (scipy.signal.resample_poly)."""
    target_fs = target_fs or load_config()["data"]["target_fs"]
    if fs <= 0:
        raise ValueError(f"sampling rate must be positive, got {fs}")
    if abs(fs - target_fs) < 1e-9:
        return np.asarray(signal, dtype=np.float64)
    ratio = Fraction(int(round(target_fs)), int(round(fs))).limit_denominator(1000)
    return resample_poly(np.asarray(signal, dtype=np.float64), ratio.numerator, ratio.denominator)


def load_cwru_recording(entry, raw_dir: Path | None = None) -> tuple[np.ndarray, float, MatSignal]:
    """Load a catalog entry, resampled to the target rate. Returns (signal, rpm, raw MatSignal).

    RPM: value stored in the file when present, otherwise the approximate RPM from the CWRU table.
    """
    raw_dir = raw_dir or path("raw_dir")
    ms = read_cwru_mat(raw_dir / entry.filename, entry.file_number)
    sig = resample_to(ms.signal, entry.fs)
    rpm = ms.rpm if ms.rpm and ms.rpm > 0 else float(entry.rpm)
    return sig, rpm, ms


# ---------------------------------------------------------------- user uploads
SUPPORTED_UPLOADS = (".mat", ".csv", ".npy")


def load_upload(source: str | Path | BinaryIO | bytes, filename: str) -> np.ndarray:
    """Read a user-supplied signal file (at its original rate; the caller resamples).

    .mat : a '*_DE_time' variable, or else the only 1-D numeric array in the file
    .csv : a column named like signal/DE/vibration/acc, or else the only numeric column
    .npy : a 1-D numeric array (or N x 1)
    """
    ext = Path(filename).suffix.lower()
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    if ext == ".mat":
        try:
            return read_cwru_mat(source).signal
        except ValueError as exc:
            if hasattr(source, "seek"):
                source.seek(0)
            mat = scipy.io.loadmat(source)
            arrays = {k: np.asarray(v) for k, v in mat.items()
                      if not k.startswith("__") and np.issubdtype(np.asarray(v).dtype, np.number)}
            vectors = {k: v.ravel() for k, v in arrays.items() if v.size > 1 and min(v.shape) == 1}
            if len(vectors) == 1:
                return next(iter(vectors.values())).astype(np.float64)
            raise ValueError(f"could not find a single vibration signal in the .mat file ({exc})") from None
    if ext == ".npy":
        arr = np.load(source, allow_pickle=False)
        arr = np.asarray(arr)
        if arr.ndim == 2 and 1 in arr.shape:
            arr = arr.ravel()
        if arr.ndim != 1 or not np.issubdtype(arr.dtype, np.number):
            raise ValueError(f".npy must contain a 1-D numeric array, got shape {arr.shape} {arr.dtype}")
        return arr.astype(np.float64)
    if ext == ".csv":
        df = pd.read_csv(source)
        numeric = df.select_dtypes(include="number")
        if numeric.shape[1] == 0:
            # maybe the file has no header row: re-read without one
            if hasattr(source, "seek"):
                source.seek(0)
            numeric = pd.read_csv(source, header=None).select_dtypes(include="number")
        if numeric.shape[1] == 0:
            raise ValueError("CSV file has no numeric column")
        preferred = [c for c in numeric.columns if re.search(r"signal|de|vib|acc", str(c), re.I)]
        if len(preferred) == 1:
            col = preferred[0]
        elif numeric.shape[1] == 1:
            col = numeric.columns[0]
        else:
            raise ValueError(f"CSV has several numeric columns {list(numeric.columns)}; "
                             "keep one, or name it 'signal'")
        return numeric[col].to_numpy(dtype=np.float64)
    raise ValueError(f"unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_UPLOADS)}")

"""Split a long signal into short, overlapping windows."""
from __future__ import annotations

import numpy as np

from pdm.config import load_config


def window_signal(signal: np.ndarray, size: int | None = None, overlap: float | None = None) -> np.ndarray:
    """Return an array of shape (n_windows, size). A trailing partial window is dropped.

    Example: 10 000 samples, size 2048, overlap 0.5 -> step 1024 -> 8 windows.
    """
    cfg = load_config()["signal"]
    size = int(size or cfg["window_size"])
    overlap = cfg["overlap"] if overlap is None else overlap
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    x = np.asarray(signal, dtype=np.float64).ravel()
    if x.size < size:
        raise ValueError(f"signal has {x.size} samples; need at least one window of {size}")
    step = max(1, int(round(size * (1 - overlap))))
    n = 1 + (x.size - size) // step
    idx = np.arange(size)[None, :] + step * np.arange(n)[:, None]
    return x[idx]

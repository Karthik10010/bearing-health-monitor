"""Per-window features: time domain, frequency domain (FFT) and envelope domain.

This module depends ONLY on NumPy and SciPy (no project imports at module level) so it can be
copied as-is to an edge device. All tunable parameters are passed in via a plain dict;
`params_from_config()` builds that dict from config.yaml.

Feature names are prefixed by group: ``t_`` time, ``f_`` frequency, ``e_`` envelope.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.stats import kurtosis, skew

_EPS = 1e-12
DEFECTS = ("bpfo", "bpfi", "bsf")
TIME_FEATURES = ["t_mean", "t_std", "t_rms", "t_min", "t_max", "t_p2p", "t_kurtosis",
                 "t_skewness", "t_crest", "t_impulse", "t_shape"]


def params_from_config() -> dict:
    from pdm.config import load_config  # imported lazily so edge copies don't need it
    cfg = load_config()
    s, b = cfg["signal"], cfg["bearing"]
    return {
        "n_bands": s["n_bands"], "band_max_hz": s["band_max_hz"],
        "envelope_band_hz": tuple(s["envelope_band_hz"]), "envelope_nfft": s["envelope_nfft"],
        "envelope_max_hz": s["envelope_max_hz"], "envelope_tolerance": s["envelope_tolerance"],
        "envelope_harmonics": s["envelope_harmonics"],
        "multipliers": {"bpfo": b["bpfo"], "bpfi": b["bpfi"], "bsf": b["bsf"]},
    }


RATIO_FEATURES = ["e_share_bpfo", "e_share_bpfi", "e_share_bsf",
                  "e_logratio_bpfo_bpfi", "e_logratio_bpfo_bsf", "e_logratio_bpfi_bsf"]


def feature_names(params: dict) -> list[str]:
    f = ["f_centroid", "f_dominant"] + [f"f_band_{i + 1}" for i in range(params["n_bands"])]
    e = [f"e_{d}_{h}" for d in DEFECTS for h in range(1, params["envelope_harmonics"] + 1)] + ["e_shaft_1"]
    return TIME_FEATURES + f + e + RATIO_FEATURES


def feature_groups(names: list[str]) -> dict[str, list[str]]:
    return {
        "time": [n for n in names if n.startswith("t_")],
        "frequency": [n for n in names if n.startswith("f_")],
        "envelope": [n for n in names if n.startswith("e_")],
        "all": list(names),
    }


# ------------------------------------------------------------------ time domain
def time_features(w: np.ndarray) -> np.ndarray:
    """w: (n_windows, n_samples) -> (n_windows, 11)."""
    mean = w.mean(axis=1)
    std = w.std(axis=1)
    rms = np.sqrt(np.mean(w ** 2, axis=1))
    mn, mx = w.min(axis=1), w.max(axis=1)
    peak = np.max(np.abs(w), axis=1)
    abs_mean = np.mean(np.abs(w), axis=1)
    kurt = kurtosis(w, axis=1, fisher=False)  # Gaussian noise ~ 3; impulsive faults >> 3
    sk = skew(w, axis=1)
    crest = peak / (rms + _EPS)
    impulse = peak / (abs_mean + _EPS)
    shape = rms / (abs_mean + _EPS)
    return np.column_stack([mean, std, rms, mn, mx, mx - mn, kurt, sk, crest, impulse, shape])


# ------------------------------------------------------------- frequency domain
def frequency_features(w: np.ndarray, fs: float, params: dict) -> np.ndarray:
    n = w.shape[1]
    x = (w - w.mean(axis=1, keepdims=True)) * np.hanning(n)[None, :]
    power = np.abs(np.fft.rfft(x, axis=1)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    total = power.sum(axis=1) + _EPS
    centroid = (power * freqs[None, :]).sum(axis=1) / total
    dominant = freqs[np.argmax(power[:, 1:], axis=1) + 1]  # skip the DC bin
    edges = np.linspace(0.0, params["band_max_hz"], params["n_bands"] + 1)
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (freqs >= lo) & (freqs < hi if hi < edges[-1] else freqs <= hi)
        bands.append(power[:, mask].sum(axis=1) / total)  # fraction of total energy
    return np.column_stack([centroid, dominant, *bands])


# -------------------------------------------------------------- envelope domain
def envelope_spectrum(w: np.ndarray, fs: float, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Band-pass -> Hilbert envelope -> magnitude spectrum. Returns (freqs, mags (n_windows, n_freqs))."""
    lo, hi = params["envelope_band_hz"]
    hi = min(hi, 0.45 * fs)
    sos = butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    xb = sosfiltfilt(sos, w, axis=1)
    env = np.abs(hilbert(xb, axis=1))
    env = env - env.mean(axis=1, keepdims=True)
    env = env * np.hanning(w.shape[1])[None, :]
    nfft = max(params["envelope_nfft"], w.shape[1])
    mags = np.abs(np.fft.rfft(env, n=nfft, axis=1))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    return freqs, mags


def envelope_features(w: np.ndarray, fs: float, rpm: float, params: dict) -> np.ndarray:
    """Peak envelope amplitude near each defect frequency (and harmonics), divided by the
    median 'background' level of the envelope spectrum. ~1 means 'nothing special there';
    a value of 10 means a peak 10x above the background."""
    freqs, mags = envelope_spectrum(w, fs, params)
    df = freqs[1] - freqs[0]
    shaft = rpm / 60.0
    bg_mask = (freqs >= 5.0) & (freqs <= params["envelope_max_hz"])
    background = np.median(mags[:, bg_mask], axis=1) + _EPS

    def peak_ratio(f0: float) -> np.ndarray:
        tol = max(params["envelope_tolerance"] * f0, 1.5 * df)
        m = (freqs >= f0 - tol) & (freqs <= f0 + tol)
        if not m.any():
            return np.zeros(w.shape[0])
        return mags[:, m].max(axis=1) / background

    cols = []
    for d in DEFECTS:
        f_def = params["multipliers"][d] * shaft
        for h in range(1, params["envelope_harmonics"] + 1):
            cols.append(peak_ratio(h * f_def))
    cols.append(peak_ratio(shaft))
    return np.column_stack(cols)


def knock_ratio_features(env: np.ndarray, harmonics: int) -> np.ndarray:
    """'Which knock dominates?' features from the envelope peak ratios (scale-free).

    share_x = (sum of x's harmonic peaks) / (sum over BPFO, BPFI, BSF); log-ratios compare the
    1x peaks (BSF uses the larger of its 1x / 2x peak, since ball faults often show at 2x BSF).
    """
    h = harmonics
    s = {d: env[:, i * h:(i + 1) * h].sum(axis=1) for i, d in enumerate(DEFECTS)}
    tot = s["bpfo"] + s["bpfi"] + s["bsf"] + _EPS
    o1, i1 = env[:, 0], env[:, h]
    b12 = np.maximum(env[:, 2 * h], env[:, 2 * h + 1])
    return np.column_stack([s["bpfo"] / tot, s["bpfi"] / tot, s["bsf"] / tot,
                            np.log((o1 + _EPS) / (i1 + _EPS)), np.log((o1 + _EPS) / (b12 + _EPS)),
                            np.log((i1 + _EPS) / (b12 + _EPS))])


# --------------------------------------------------------------------- all
def extract_features(windows: np.ndarray, fs: float, rpm: float, params: dict | None = None) -> np.ndarray:
    """windows: (n_windows, n_samples) -> feature matrix (n_windows, n_features), see feature_names()."""
    params = params or params_from_config()
    w = np.atleast_2d(np.asarray(windows, dtype=np.float64))
    env = envelope_features(w, fs, rpm, params)
    return np.hstack([
        time_features(w),
        frequency_features(w, fs, params),
        env,
        knock_ratio_features(env, params["envelope_harmonics"]),
    ])

"""Step 2: load every CWRU recording, resample to 12 kHz, window it, extract features.

Outputs
  data/features/features.csv           one row per window (+ recording metadata)
  reports/metrics/data_checks.json     per-recording notes (keys used, RPM, sample-rate check)
  reports/figures/feature_sanity.png   faults vs. kurtosis / envelope features
"""
import _bootstrap  # noqa: F401

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pdm.config import load_config, path
from pdm.data.cwru_catalog import CATALOG, CLASSES
from pdm.data.loader import load_cwru_recording
from pdm.signal.features import extract_features, feature_names, params_from_config
from pdm.signal.windowing import window_signal

log = logging.getLogger("build_features")


def check_sample_rate(raw: np.ndarray, fs: float, rpm: float) -> dict:
    """Sanity check of the assumed sampling rate: with the right fs, the 1x shaft-speed
    peak sits at rpm/60 Hz. Under a wrong fs (e.g. 12k vs 48k) it would appear 4x off."""
    n = min(raw.size, int(fs * 4))
    x = raw[:n] - raw[:n].mean()
    spec = np.abs(np.fft.rfft(x * np.hanning(n)))
    f = np.fft.rfftfreq(n, 1 / fs)
    shaft = rpm / 60
    band = (f > 0.8 * shaft) & (f < 1.2 * shaft)
    ref = (f > 5) & (f < 200)
    peak_f = float(f[band][np.argmax(spec[band])])
    prominence = float(spec[band].max() / np.median(spec[ref]))
    return {"assumed_fs": fs, "shaft_hz_expected": round(shaft, 2), "shaft_peak_found_hz": round(peak_f, 2),
            "peak_prominence": round(prominence, 1),
            "consistent": bool(abs(peak_f - shaft) < 0.03 * shaft and prominence > 5)}


def main() -> None:
    params = params_from_config()
    names = feature_names(params)
    target_fs = load_config()["data"]["target_fs"]
    frames, checks = [], []
    for e in CATALOG:
        sig, rpm, ms = load_cwru_recording(e)
        win = window_signal(sig)
        X = extract_features(win, target_fs, rpm, params)
        df = pd.DataFrame(X, columns=names)
        df.insert(0, "window_idx", np.arange(len(df)))
        for col, val in reversed(list({
            "recording_id": e.recording_id, "file_number": e.file_number, "fault": e.fault,
            "size_in": e.size_in, "load_hp": e.load_hp, "rpm": rpm}.items())):
            df.insert(0, col, val)
        frames.append(df)
        chk = {"recording_id": e.recording_id, "file_number": e.file_number, "key_used": ms.key,
               "rpm_in_file": ms.rpm, "rpm_used": rpm, "original_fs": e.fs,
               "original_samples": int(ms.signal.size), "seconds": round(ms.signal.size / e.fs, 2),
               "windows": len(df), "notes": ms.notes}
        if e.fault == "Normal":
            chk["sample_rate_check"] = check_sample_rate(ms.signal, e.fs, rpm)
        checks.append(chk)
        log.info("%-12s %5d windows  rpm=%.0f  %s", e.recording_id, len(df), rpm, "; ".join(ms.notes))

    feats = pd.concat(frames, ignore_index=True)
    out = path("features_csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(out, index=False)

    metrics_dir = path("metrics_dir")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "data_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")

    # ---- sanity plot
    fig_dir = path("figures_dir")
    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = {"Normal": "#2a9d8f", "Inner race": "#e76f51", "Outer race": "#264653", "Ball": "#e9c46a"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    pairs = [("t_kurtosis", "t_rms"), ("e_bpfo_1", "e_bpfi_1"), ("e_bsf_2", "t_crest")]
    sample = feats.groupby("recording_id", group_keys=False).apply(lambda g: g.sample(min(len(g), 60), random_state=0))
    for ax, (fx, fy) in zip(axes, pairs):
        for c in CLASSES:
            s = sample[sample.fault == c]
            ax.scatter(s[fx], s[fy], s=6, alpha=0.5, label=c, color=colors[c])
        ax.set_xlabel(fx); ax.set_ylabel(fy)
        log_axes = {"t_kurtosis", "t_rms", "e_bpfo_1", "e_bpfi_1"}  # wide-range features only
        if fx in log_axes:
            ax.set_xscale("log")
        if fy in log_axes:
            ax.set_yscale("log")
    axes[0].legend(markerscale=3)
    fig.suptitle("CWRU windows (60 per recording): faults vs. key features")
    fig.tight_layout()
    fig.savefig(fig_dir / "feature_sanity.png", dpi=130)

    summary = feats.groupby("fault")[["t_kurtosis", "t_rms", "e_bpfo_1", "e_bpfi_1", "e_bsf_2"]].median()
    print(f"\nWrote {out} : {len(feats)} windows x {len(names)} features from {feats.recording_id.nunique()} recordings")
    print(feats.groupby("fault").agg(recordings=("recording_id", "nunique"), windows=("window_idx", "size")))
    print("\nMedian feature values per class:\n", summary.round(2))
    print("\nNormal-file sample-rate checks:")
    for c in checks:
        if "sample_rate_check" in c:
            print(" ", c["recording_id"], c["sample_rate_check"])


if __name__ == "__main__":
    main()

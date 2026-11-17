"""Standalone edge inference: vibration file -> fault diagnosis.

Needs ONLY numpy, scipy and onnxruntime (no scikit-learn, no project package). Copy this whole
`edge/` folder to the device:  edge_infer.py, edge_features.py, edge_config.json, *.onnx

Usage:
    python edge_infer.py SIGNAL_FILE --fs 12000 --rpm 1750 [--json]

SIGNAL_FILE: .npy (1-D array), .csv (one numeric column) or .mat (a '*_DE_time' variable or a
single vector). The signal is resampled to the model's rate (12 kHz) with an anti-aliasing filter.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import onnxruntime as ort
import scipy.io
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from edge_features import extract_features  # noqa: E402  (copied from pdm/signal/features.py)


def read_signal(p: Path) -> np.ndarray:
    ext = p.suffix.lower()
    if ext == ".npy":
        return np.load(p, allow_pickle=False).astype(np.float64).ravel()
    if ext == ".csv":
        rows = []
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(float(line.split(",")[0]))
            except ValueError:
                continue  # header or blank line
        return np.asarray(rows)
    if ext == ".mat":
        mat = scipy.io.loadmat(p)
        de = [k for k in mat if re.match(r"^X\d+_DE_time$", k)]
        if len(de) > 1:
            num = re.sub(r"\D", "", p.stem)
            de = [k for k in de if k == f"X{int(num):03d}_DE_time"] if num else []
        if len(de) == 1:
            return np.asarray(mat[de[0]], dtype=np.float64).ravel()
        vecs = [np.asarray(v).ravel() for k, v in mat.items()
                if not k.startswith("__") and np.asarray(v).dtype.kind == "f" and min(np.asarray(v).shape) == 1]
        if len(vecs) == 1:
            return vecs[0].astype(np.float64)
        raise ValueError("could not find a single vibration signal in the .mat file")
    raise ValueError(f"unsupported file type {ext}")


def resample(x: np.ndarray, fs: float, target: float) -> np.ndarray:
    if abs(fs - target) < 1e-9:
        return x
    r = Fraction(int(round(target)), int(round(fs))).limit_denominator(1000)
    return resample_poly(x, r.numerator, r.denominator)


def windows(x: np.ndarray, size: int, overlap: float) -> np.ndarray:
    step = max(1, int(round(size * (1 - overlap))))
    if x.size < size:
        raise ValueError(f"signal too short: {x.size} samples < window {size}")
    n = 1 + (x.size - size) // step
    return x[np.arange(size)[None, :] + step * np.arange(n)[:, None]]


def knock_call(X_all: np.ndarray, names: list, rules: dict):
    """Clear knock signature check (same rule as pdm/physics.py): the strongest family's envelope
    peak must be >= min_peak_ratio x background and >= min_dominance x the next family."""
    med = {k: float(np.median(X_all[:, names.index(k)])) for k in ("e_bpfo_1", "e_bpfi_1", "e_bsf_1", "e_bsf_2")}
    peaks = {"Outer race": med["e_bpfo_1"], "Inner race": med["e_bpfi_1"], "Ball": max(med["e_bsf_1"], med["e_bsf_2"])}
    order = sorted(peaks, key=peaks.get, reverse=True)
    top, second = peaks[order[0]], peaks[order[1]]
    if top >= rules["min_peak_ratio"] and (second <= 0 or top / second >= rules["min_dominance"]):
        return order[0]
    return None


def run(sess, X: np.ndarray) -> np.ndarray:
    return sess.run(None, {sess.get_inputs()[0].name: X.astype(np.float32)})[1]


def diagnose(signal: np.ndarray, fs: float, rpm: float, cfg: dict, sessions: dict) -> dict:
    x = resample(signal, fs, cfg["target_fs"])
    W = windows(x, cfg["window_size"], cfg["overlap"])
    X_all = extract_features(W, cfg["target_fs"], rpm, cfg["feature_params"])
    names = cfg["all_feature_names"]
    X = X_all[:, [names.index(c) for c in cfg["feature_cols"]]]
    P = run(sessions["fault"], X).mean(axis=0)
    classes = cfg["classes"]["fault"]
    ml = classes[int(P.argmax())]
    knock = knock_call(X_all, names, cfg["physics"])
    final = knock or ml
    k = classes.index(final)
    out = {"fault": final, "confidence": round(float(P[k]), 4), "ml_fault": ml,
           "decision_source": "knock signature (physics)" if knock else "ML model",
           "probabilities": {c: round(float(p), 4) for c, p in zip(classes, P)}, "windows": int(len(W))}
    if final != "Normal" and "severity" in sessions:
        Xs = X_all[:, [names.index(c) for c in cfg["severity_cols"]]]
        SP = run(sessions["severity"], Xs).mean(axis=0)
        s_classes = cfg["classes"]["severity"]
        j = int(SP.argmax())
        out["size_in"] = float(s_classes[j])
        out["size_confidence"] = round(float(SP[j]), 4)
    return out


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("signal_file", type=Path)
    ap.add_argument("--fs", type=float, required=True, help="sampling rate of the file (Hz)")
    ap.add_argument("--rpm", type=float, required=True, help="shaft speed (rev/min)")
    ap.add_argument("--fault-model", default=None, help="override the fault ONNX file (default from edge_config.json)")
    ap.add_argument("--json", action="store_true", help="print machine-readable JSON only")
    a = ap.parse_args(argv)

    cfg = json.loads((HERE / "edge_config.json").read_text(encoding="utf-8"))
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    files = {"fault": a.fault_model or cfg["models"]["fault"], "severity": cfg["models"].get("severity")}
    sessions = {k: ort.InferenceSession(str(HERE / f), so, providers=["CPUExecutionProvider"])
                for k, f in files.items() if f}
    res = diagnose(read_signal(a.signal_file), a.fs, a.rpm, cfg, sessions)
    if a.json:
        print(json.dumps(res))
    else:
        print(f"Diagnosis : {res['fault']}  (decided by {res['decision_source']}; ML vote {res['confidence']:.0%}, "
              f"{res['windows']} windows)")
        if "size_in" in res:
            print(f'Est. size : {res["size_in"]:.3f}"  (confidence {res["size_confidence"]:.0%})')
        print("Probabilities:", ", ".join(f"{c} {p:.0%}" for c, p in res["probabilities"].items()))
    return res


if __name__ == "__main__":
    main()

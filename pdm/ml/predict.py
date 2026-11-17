"""Recording-level prediction with the frozen fault + severity models (Phase 2 / runtime)."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from pdm.ml.train import load_model


@dataclass
class RecordingPrediction:
    fault: str
    confidence: float                   # vote share of the winning class, averaged over windows
    probabilities: dict[str, float]     # averaged over all windows
    window_predictions: list[str]       # per-window class (for the timeline plot)
    window_agreement: float             # share of windows that agree with the final class
    size_in: float | None               # estimated fault diameter (inches); None if Normal
    size_probabilities: dict[str, float] | None
    size_confidence: float | None

    def to_dict(self) -> dict:
        return asdict(self)


class Predictor:
    def severity_estimate(self, X: np.ndarray) -> tuple[float, dict, float]:
        """Fault size from the severity RF (X already holds the model's feature columns)."""
        sm = self.severity["model"]
        SP = sm.predict_proba(X).mean(axis=0)
        s_classes = [str(c) for c in sm.classes_]
        j = int(SP.argmax())
        return float(s_classes[j]), {c: float(p) for c, p in zip(s_classes, SP)}, float(SP[j])

    def severity_for(self, X: np.ndarray, names: list[str]) -> tuple[float, dict, float]:
        idx = {n: i for i, n in enumerate(names)}
        return self.severity_estimate(np.atleast_2d(X)[:, [idx[c] for c in self.severity_cols]])

    def __init__(self, fault_bundle: dict | None = None, severity_bundle: dict | None = None):
        self.fault = fault_bundle or load_model("fault_rf")
        self.severity = severity_bundle or load_model("severity_rf")
        self.feature_cols: list[str] = self.fault["feature_cols"]
        self.severity_cols: list[str] = self.severity["feature_cols"]

    def predict(self, X: np.ndarray, names: list[str] | None = None) -> RecordingPrediction:
        """X: windows x features. If `names` is given, X may hold extra columns; the model's own
        feature columns are selected by name."""
        X = np.atleast_2d(X)
        Xs = X
        if names is not None:
            idx = {n: i for i, n in enumerate(names)}
            Xs = X[:, [idx[c] for c in self.severity_cols]]
            X = X[:, [idx[c] for c in self.feature_cols]]
        if X.shape[1] != len(self.feature_cols):
            raise ValueError(f"expected {len(self.feature_cols)} features, got {X.shape[1]}")
        fm = self.fault["model"]
        P = fm.predict_proba(X)
        classes = [str(c) for c in fm.classes_]
        mean_p = P.mean(axis=0)
        k = int(mean_p.argmax())
        fault = classes[k]
        win_pred = [classes[i] for i in P.argmax(axis=1)]

        size = size_p = size_conf = None
        if fault != "Normal":
            size, size_p, size_conf = self.severity_estimate(Xs)

        return RecordingPrediction(
            fault=fault, confidence=float(mean_p[k]),
            probabilities={c: float(p) for c, p in zip(classes, mean_p)},
            window_predictions=win_pred,
            window_agreement=float(np.mean([w == fault for w in win_pred])),
            size_in=size, size_probabilities=size_p, size_confidence=size_conf,
        )

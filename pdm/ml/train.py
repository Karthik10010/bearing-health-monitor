"""Random Forest training helpers (fault type + fault size)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from pdm.config import load_config, path
from pdm.signal.features import feature_names, params_from_config

META_COLS = ["recording_id", "file_number", "fault", "size_in", "load_hp", "rpm", "window_idx"]


def size_label(size_in: float) -> str:
    return f"{size_in:.3f}"


def load_features(csv: str | Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv or path("features_csv"))
    df["size_label"] = df["size_in"].map(size_label)
    return df


def computed_feature_cols() -> list[str]:
    """Every feature extracted per window (also used for display and explanations)."""
    return feature_names(params_from_config())


def all_feature_cols(kind: str = "fault_rf") -> list[str]:
    """Features a model uses: everything computed minus its exclude list
    (ml.<kind>.exclude_features if set, else ml.exclude_features)."""
    ml = load_config()["ml"]
    excl = set(ml[kind].get("exclude_features", ml.get("exclude_features", [])))
    return [c for c in computed_feature_cols() if c not in excl]


def make_rf(kind: str = "fault_rf", **overrides) -> RandomForestClassifier:
    cfg = load_config()["ml"]
    p = {**{k: v for k, v in cfg[kind].items() if k != "exclude_features"}, **overrides}
    return RandomForestClassifier(
        n_estimators=p["n_estimators"], max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"],
        class_weight=p["class_weight"], random_state=cfg["random_state"], n_jobs=-1,
    )


def faulty_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["fault"] != "Normal"]


def fit_fault_model(df: pd.DataFrame, cols: list[str] | None = None, **overrides) -> RandomForestClassifier:
    cols = cols or all_feature_cols()
    return make_rf("fault_rf", **overrides).fit(df[cols].to_numpy(), df["fault"].to_numpy())


def fit_severity_model(df: pd.DataFrame, cols: list[str] | None = None, **overrides) -> RandomForestClassifier:
    cols = cols or all_feature_cols("severity_rf")
    d = faulty_only(df)
    return make_rf("severity_rf", **overrides).fit(d[cols].to_numpy(), d["size_label"].to_numpy())


def save_model(model, name: str, feature_cols: list[str], extra: dict | None = None) -> Path:
    """Save as models/<name>.joblib together with a small JSON 'model card'."""
    models_dir = path("models_dir")
    models_dir.mkdir(parents=True, exist_ok=True)
    out = models_dir / f"{name}.joblib"
    joblib.dump({"model": model, "feature_cols": feature_cols, "classes": list(model.classes_)}, out)
    card = {
        "name": name, "file": out.name, "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "classes": [str(c) for c in model.classes_], "n_features": len(feature_cols),
        "n_estimators": model.n_estimators, "max_depth": model.max_depth,
        "feature_params": params_from_config(), **(extra or {}),
    }
    (models_dir / f"{name}.json").write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")
    return out


def load_model(name: str) -> dict:
    # joblib uses pickle: only ever load models this project trained itself (models/ folder).
    # Never point this at a user-uploaded or downloaded .joblib file. (ONNX is the portable,
    # non-executable format used on the edge side.)
    return joblib.load(path("models_dir") / f"{name}.joblib")


def recording_probabilities(proba: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average window probabilities per recording -> (unique_groups, mean_proba)."""
    uniq, inv = np.unique(groups, return_inverse=True)
    sums = np.zeros((len(uniq), proba.shape[1]))
    np.add.at(sums, inv, proba)
    counts = np.bincount(inv)[:, None]
    return uniq, sums / counts

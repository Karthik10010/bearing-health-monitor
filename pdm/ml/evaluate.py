"""Evaluation: leave-one-load-out (LOLO), naive random split, feature-group comparison.

LOLO: train on 3 loads, test on the 4th, repeat for each load. Every recording is entirely in
train OR test, so the model is always examined on recordings it has never seen.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from pdm.ml.train import make_rf, recording_probabilities


def _metrics(y_true, y_pred, labels) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": {str(c): {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, c in enumerate(labels)},
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def _mean_std(folds: list[dict], key: str) -> dict:
    v = np.array([f[key] for f in folds])
    return {"mean": float(v.mean()), "std": float(v.std()), "per_fold": v.round(4).tolist()}


def lolo_evaluate(df: pd.DataFrame, cols: list[str], target: str, labels: list,
                  kind: str = "fault_rf", rf_overrides: dict | None = None,
                  keep_oof: bool = True, fold_col: str = "load_hp") -> dict:
    """Leave-one-group-out evaluation at window AND recording level.

    Default ``fold_col="load_hp"`` = leave-one-load-out. Any other column can define the folds
    (e.g. leave-one-fault-size-out); every recording must belong to exactly one fold.

    Returns summary metrics (mean +/- std over the 4 folds), summed confusion matrices, and
    (optionally) the out-of-fold window probabilities used later by the RL simulator.
    """
    rf_overrides = rf_overrides or {}
    labels = list(labels)
    win_folds, rec_folds, oof = [], [], []
    cm_win = np.zeros((len(labels), len(labels)), dtype=int)
    cm_rec = np.zeros_like(cm_win)
    for load in sorted(df[fold_col].unique()):
        tr, te = df[df[fold_col] != load], df[df[fold_col] == load]
        if te.empty or tr.empty:
            continue
        model = make_rf(kind, **rf_overrides).fit(tr[cols].to_numpy(), tr[target].to_numpy())
        proba = model.predict_proba(te[cols].to_numpy())
        # align columns to `labels` (a class can be missing from a training fold)
        P = np.zeros((len(te), len(labels)))
        for j, c in enumerate(model.classes_):
            P[:, labels.index(c)] = proba[:, j]
        pred = np.array(labels, dtype=object)[P.argmax(axis=1)]
        m = _metrics(te[target].to_numpy(), pred, labels)
        m["test_fold"] = load if isinstance(load, str) else int(load)
        win_folds.append(m)
        cm_win += np.array(m["confusion"])

        rec_ids, rec_P = recording_probabilities(P, te["recording_id"].to_numpy())
        truth = te.groupby("recording_id")[target].first().loc[rec_ids].to_numpy()
        rec_pred = np.array(labels, dtype=object)[rec_P.argmax(axis=1)]
        mr = _metrics(truth, rec_pred, labels)
        mr["test_fold"] = m["test_fold"]
        mr["recordings"] = [{"recording_id": str(r), "true": str(t), "pred": str(p), "confidence": float(c.max())}
                            for r, t, p, c in zip(rec_ids, truth, rec_pred, rec_P)]
        rec_folds.append(mr)
        cm_rec += np.array(mr["confusion"])

        if keep_oof:
            keep = ["recording_id", "window_idx", "fault", "size_label", "load_hp"]
            keep += [c for c in ("e_bpfo_1", "e_bpfi_1", "e_bsf_1", "e_bsf_2") if c in te.columns]  # physics check
            o = te[keep].copy()
            for j, c in enumerate(labels):
                o[f"p_{c}"] = P[:, j]
            o["pred"] = pred
            oof.append(o)

    out = {
        "labels": [str(l) for l in labels],
        "window": {"accuracy": _mean_std(win_folds, "accuracy"), "macro_f1": _mean_std(win_folds, "macro_f1"),
                   "per_class_f1": {str(c): _mean_std([{"v": f["per_class"][str(c)]["f1"]} for f in win_folds], "v")
                                    for c in labels},
                   "confusion": cm_win.tolist(), "folds": win_folds},
        "recording": {"accuracy": _mean_std(rec_folds, "accuracy"), "macro_f1": _mean_std(rec_folds, "macro_f1"),
                      "confusion": cm_rec.tolist(), "folds": rec_folds},
    }
    if keep_oof:
        out["oof"] = pd.concat(oof, ignore_index=True)
    return out


def naive_split_evaluate(df: pd.DataFrame, cols: list[str], target: str, labels: list,
                         kind: str = "fault_rf", seed: int = 0) -> dict:
    """The WRONG way (for comparison): shuffle all windows and split 80/20. Overlapping windows
    from the same recording land on both sides, so the 'exam' contains near-copies of the homework."""
    tr, te = train_test_split(df, test_size=0.2, random_state=seed, stratify=df[target])
    model = make_rf(kind).fit(tr[cols].to_numpy(), tr[target].to_numpy())
    m = _metrics(te[target].to_numpy(), model.predict(te[cols].to_numpy()), list(labels))
    return {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}


def feature_group_comparison(df: pd.DataFrame, groups: dict[str, list[str]], target: str, labels: list,
                             kind: str = "fault_rf", fold_col: str = "load_hp") -> dict:
    res = {}
    for name, cols in groups.items():
        r = lolo_evaluate(df, cols, target, labels, kind, keep_oof=False, fold_col=fold_col)
        res[name] = {"n_features": len(cols), "window_macro_f1": r["window"]["macro_f1"],
                     "recording_accuracy": r["recording"]["accuracy"]}
    return res


def size_fold_column(df: pd.DataFrame) -> pd.Series:
    """Folds for leave-one-fault-size-out: fold k holds every fault recording of the k-th size
    (0.007/0.014/0.021/0.028) plus the Normal recording at load k, so each fold tests on bearings
    (physical specimens) the model has never seen at any load."""
    sizes = ["0.007", "0.014", "0.021", "0.028"]
    return df.apply(lambda r: int(r["load_hp"]) if r["fault"] == "Normal" else sizes.index(r["size_label"]),
                    axis=1)


def normalise_rows(cm) -> list[list[float]]:
    a = np.asarray(cm, dtype=float)
    s = a.sum(axis=1, keepdims=True)
    return np.divide(a, s, out=np.zeros_like(a), where=s > 0).round(4).tolist()

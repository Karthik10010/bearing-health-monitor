"""Module 2 (fault RF evaluation) and Module 3 (health rules) tests on synthetic data."""
import numpy as np
import pandas as pd
import pytest

from pdm.health import assess_health
from pdm.ml.evaluate import lolo_evaluate, naive_split_evaluate, normalise_rows
from pdm.ml.predict import Predictor
from pdm.ml.train import make_rf, recording_probabilities

RULES = {"healthy_min_confidence": 0.70, "low_confidence_threshold": 0.60, "warning_sizes": [0.007, 0.014],
         "critical_min_size": 0.021, "escalate_size": 0.014, "escalate_min_load_hp": 2}
CLASSES = ["Normal", "Inner race", "Outer race", "Ball"]


# ------------------------------------------------------------------ health rules (every row)
def test_healthy():
    h = assess_health("Normal", 0.92, None, 1, RULES)
    assert h.status == "Healthy" and not h.low_confidence and h.label == "Rule-based estimate"


def test_normal_but_unsure_is_not_healthy():
    h = assess_health("Normal", 0.65, None, 1, RULES)
    assert h.status == "Warning" and not h.low_confidence


def test_warning_early():
    assert assess_health("Outer race", 0.9, 0.007, 3, RULES).status == "Warning"


def test_moderate_low_load_warning_high_load_critical():
    assert assess_health("Ball", 0.9, 0.014, 1, RULES).status == "Warning"
    assert assess_health("Ball", 0.9, 0.014, 2, RULES).status == "Critical"


@pytest.mark.parametrize("size", [0.021, 0.028])
def test_critical_large(size):
    assert assess_health("Inner race", 0.95, size, 0, RULES).status == "Critical"


def test_low_confidence_flag_any_class():
    h = assess_health("Inner race", 0.45, 0.007, 0, RULES)
    assert h.low_confidence and any("re-measure" in r for r in h.reasons)
    assert assess_health("Normal", 0.5, None, 0, RULES).low_confidence


def test_fault_without_size_raises():
    with pytest.raises(ValueError):
        assess_health("Ball", 0.9, None, 0, RULES)


# ------------------------------------------------------------------ evaluation mechanics
def _synthetic_df(n_win=30, seed=0):
    """4 classes x 4 loads, one 'recording' per (class, load); class shifts feature means."""
    rng = np.random.default_rng(seed)
    rows = []
    for ci, c in enumerate(CLASSES):
        for load in range(4):
            X = rng.normal(loc=ci * 2.0, scale=1.0, size=(n_win, 3))
            for w, x in enumerate(X):
                rows.append({"recording_id": f"{c}_{load}", "fault": c, "size_label": "0.007" if ci else "0.000",
                             "load_hp": load, "window_idx": w, "a": x[0], "b": x[1], "c": x[2]})
    return pd.DataFrame(rows)


def test_lolo_separates_loads_and_scores_well():
    df = _synthetic_df()
    res = lolo_evaluate(df, ["a", "b", "c"], "fault", CLASSES)
    assert len(res["window"]["folds"]) == 4
    assert sorted(f["test_fold"] for f in res["window"]["folds"]) == [0, 1, 2, 3]
    assert res["window"]["macro_f1"]["mean"] > 0.8
    assert res["recording"]["accuracy"]["mean"] == 1.0
    assert np.array(res["recording"]["confusion"]).sum() == 16  # 16 recordings, each tested once
    oof = res["oof"]
    assert len(oof) == len(df)
    assert np.allclose(oof[[f"p_{c}" for c in CLASSES]].sum(axis=1), 1.0)


def test_naive_split_runs():
    r = naive_split_evaluate(_synthetic_df(), ["a", "b", "c"], "fault", CLASSES)
    assert 0 <= r["macro_f1"] <= 1


def test_recording_probabilities_average():
    P = np.array([[1, 0], [0, 1], [0.5, 0.5]])
    ids, M = recording_probabilities(P, np.array(["x", "x", "y"]))
    assert list(ids) == ["x", "y"] and np.allclose(M, [[0.5, 0.5], [0.5, 0.5]])


def test_normalise_rows_handles_empty_row():
    assert normalise_rows([[2, 2], [0, 0]]) == [[0.5, 0.5], [0.0, 0.0]]


def test_predictor_recording_level():
    df = _synthetic_df()
    cols = ["a", "b", "c"]
    fm = make_rf().fit(df[cols].to_numpy(), df["fault"].to_numpy())
    faulty = df[df.fault != "Normal"]
    sm = make_rf("severity_rf").fit(faulty[cols].to_numpy(), faulty["size_label"].to_numpy())
    pred = Predictor({"model": fm, "feature_cols": cols}, {"model": sm, "feature_cols": cols})
    out = pred.predict(df[df.recording_id == "Ball_2"][cols].to_numpy())
    assert out.fault == "Ball" and out.confidence > 0.5 and out.size_in == 0.007
    assert abs(sum(out.probabilities.values()) - 1) < 1e-9
    normal = pred.predict(df[df.recording_id == "Normal_1"][cols].to_numpy())
    assert normal.fault == "Normal" and normal.size_in is None
    with pytest.raises(ValueError):
        pred.predict(np.zeros((2, 5)))

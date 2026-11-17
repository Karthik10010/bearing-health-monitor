"""pipeline.analyze() and Module 6 (conversational AI) tests. Chat runs in fallback mode; Gemini is mocked."""
import json

import numpy as np
import pytest

from pdm.config import path
from pdm.llm import guard

pytestmark = pytest.mark.skipif(not (path("models_dir") / "ppo_agent.zip").exists(),
                                reason="train models first (scripts 03-06)")


@pytest.fixture(scope="module")
def report():
    from pdm.data.loader import load_upload
    from pdm.pipeline import analyze
    f = path("samples_dir") / "cwru_OR014at6_2.csv"
    x = load_upload(f.read_bytes(), f.name)
    return analyze(x, 12000, 1749, 2, source=f.name, whatif_rollouts=100)


# ------------------------------------------------------------------ pipeline
def test_report_structure(report):
    d = report.to_dict()
    assert "arrays" not in d
    assert set(d["diagnosis"]["probabilities"]) == {"Normal", "Inner race", "Outer race", "Ball"}
    assert abs(sum(d["diagnosis"]["probabilities"].values()) - 1) < 1e-6
    assert d["health"]["label"] == "Rule-based estimate"
    assert d["recommendation"]["label"].startswith("Trained in a SIMULATION")
    assert d["whatif_continue"]["simulated"] is True
    assert any("TRAINED on" in w for w in d["warnings"])  # sample is a training recording
    json.loads(report.to_json())  # serialisable
    assert len(report.arrays["window_predictions"]) == d["diagnosis"]["n_windows"]


def test_analyze_input_validation():
    from pdm.pipeline import analyze
    with pytest.raises(ValueError):
        analyze(np.zeros(20000), 12000, 1750, 1)
    with pytest.raises(ValueError):
        analyze(np.array([1.0, np.nan] * 5000), 12000, 1750, 1)
    r = analyze(np.random.default_rng(0).normal(size=48000) * 0.06, 48000, 2500, 5, run_whatif=False)
    assert any("RPM" in w for w in r.warnings) and any("load" in w for w in r.warnings)
    assert not any("TRAINED on" in w for w in r.warnings)


def test_pipeline_matches_independent_model_call(report):
    """No hard-coding: the report's ML opinion equals a direct call to the saved model on the same
    features, and the final class follows the documented ML + physics rule."""
    from pdm.ml.train import load_model
    from pdm.physics import knock_call_from_matrix
    bundle = load_model("fault_rf")
    names = report.arrays["feature_names"]
    X = report.arrays["features_per_window"].astype(np.float64)
    m = bundle["model"]
    p = m.predict_proba(X[:, [names.index(c) for c in bundle["feature_cols"]]]).mean(axis=0)
    assert m.classes_[p.argmax()] == report.diagnosis["ml_fault"]
    call = knock_call_from_matrix(X, names)
    assert report.diagnosis["fault"] == (call or report.diagnosis["ml_fault"])
    assert report.diagnosis["decision_source"] == ("knock signature (physics)" if call else "ML model")


def test_physics_rule_and_condition_index():
    from pdm.health import assess_health
    from pdm.physics import knock_evidence
    base = {"e_bpfo_1": 1.8, "e_bpfi_1": 1.5, "e_bsf_1": 1.4, "e_bsf_2": 1.3}
    assert knock_evidence(base).call is None                                   # healthy: silent
    assert knock_evidence({**base, "e_bpfo_1": 12.0}).call == "Outer race"     # clear BPFO knock
    assert knock_evidence({**base, "e_bpfo_1": 6.0, "e_bpfi_1": 5.0}).call is None  # not dominant
    assert knock_evidence({**base, "e_bsf_2": 7.0}).call == "Ball"            # 2x BSF counts
    idx = [assess_health(*a).condition_index for a in [("Normal", 0.95, None, 1), ("Ball", 0.9, 0.007, 1),
                                                        ("Ball", 0.9, 0.014, 1), ("Ball", 0.9, 0.014, 3),
                                                        ("Ball", 0.9, 0.021, 1), ("Ball", 0.9, 0.028, 1)]]
    assert idx == sorted(idx) and idx[0] < 33 < idx[1] < 66 < idx[3]
    h = assess_health("Outer race", 0.9, 0.007, 1, note="the methods disagree")
    assert h.low_confidence and any("disagree" in r for r in h.reasons)


# ------------------------------------------------------------------ guard
def test_guard_numbers():
    ctx = {"diagnosis": {"confidence": 0.9412, "freq": 104.62}, "p_failure_within_90_days": 0.25}
    assert guard.unsupported_numbers("94% sure, peak at 104.6 Hz, 25% in 90 days", ctx) == []
    assert guard.unsupported_numbers("confidence 87% and 3 harmonics", ctx) == ["87%"]
    assert guard.unsupported_numbers("about 1,234 hours", ctx) == ["1,234"]


# ------------------------------------------------------------------ chat (fallback for all 7 types)
@pytest.mark.parametrize("qtype", ["condition", "why_fault", "action", "why_rl", "whatif", "simple", "compare"])
def test_fallback_all_question_types(report, qtype):
    from pdm.llm.chat import answer
    from pdm.llm.fallback import QUESTION_TYPES, classify
    q = QUESTION_TYPES[qtype]
    assert classify(q) == qtype
    a = answer(q, report, use_gemini=False)
    assert a.mode == "fallback" and a.question_type == qtype and len(a.text) > 40
    assert a.unsupported_numbers == [], a.text  # every number comes from the report
    if qtype in ("action", "why_rl", "whatif"):
        assert "SIMULAT" in a.text.upper()


def test_reduce_load_whatif_runs_simulator(report):
    from pdm.llm.chat import answer
    a = answer("What if we reduce the load and keep running?", report, use_gemini=False)
    assert "whatif_reduce_load" in a.context["additional_simulations"]
    assert "REDUCED" in a.text and a.unsupported_numbers == []


def test_gemini_path_mocked(report, monkeypatch):
    from pdm.llm import chat, gemini_client
    seen = {}

    def fake_ask(question, ctx_json, history=None):
        seen["ctx"] = json.loads(ctx_json)
        return f"It is {report.diagnosis['fault']} with 12345 imaginary readings."

    monkeypatch.setattr(gemini_client, "ask", fake_ask)
    a = chat.answer("why?", report, use_gemini=True)
    assert a.mode == "gemini" and "12345" in a.unsupported_numbers
    assert "arrays" not in seen["ctx"] and "signal_12k" not in json.dumps(seen["ctx"])


def test_gemini_failure_falls_back(report, monkeypatch):
    from pdm.llm import chat, gemini_client

    def boom(*a, **k):
        raise ConnectionError("offline")

    monkeypatch.setattr(gemini_client, "ask", boom)
    a = chat.answer("What is the bearing condition?", report, use_gemini=True)
    assert a.mode == "fallback" and "Gemini unavailable" in a.note

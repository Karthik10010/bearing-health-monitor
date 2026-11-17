"""Module 4 (Edge AI) tests."""
import json
import subprocess
import sys

import numpy as np
import pytest

from pdm.config import PROJECT_ROOT, path
from pdm.edge.benchmark import time_fn
from pdm.edge.export import onnx_session, parity_check, to_onnx
from pdm.ml.train import make_rf

EDGE = PROJECT_ROOT / "edge"


def test_onnx_parity_on_synthetic_rf(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 6))
    y = np.where(X[:, 0] + X[:, 1] > 0, "A", "B")
    m = make_rf(n_estimators=15, max_depth=6).fit(X, y)
    sess = onnx_session(to_onnx(m, 6, tmp_path / "m.onnx"))
    p = parity_check(m, sess, X)
    assert p["label_agreement"] >= 0.99
    assert p["max_abs_prob_diff"] < 1e-4 or p["label_mismatches"] <= 2


def test_time_fn_fields():
    r = time_fn(lambda: sum(range(100)), repeats=20, warmup=2)
    assert r["repeats"] == 20 and 0 <= r["median_ms"] <= r["p95_ms"]


needs_bundle = pytest.mark.skipif(not (EDGE / "edge_config.json").exists() or not (path("raw_dir") / "130.mat").exists(),
                                  reason="run scripts/04_edge_benchmark.py (and 01) first")


@needs_bundle
def test_edge_infer_matches_main_pipeline_model():
    """Standalone edge_infer.py (ONNX, full model) must give the same ML opinion AND the same final
    (ML + physics) diagnosis as the main pipeline, on real CWRU recordings."""
    from pdm.data.cwru_catalog import by_number
    from pdm.data.loader import read_cwru_mat
    from pdm.pipeline import analyze

    for num in (97, 108, 130, 199, 3007):
        e = by_number(num)
        out = subprocess.run([sys.executable, str(EDGE / "edge_infer.py"), str(path("raw_dir") / e.filename),
                              "--fs", str(e.fs), "--rpm", str(e.rpm), "--fault-model", "fault_full.onnx", "--json"],
                             capture_output=True, text=True, check=True)
        edge = json.loads(out.stdout)
        sig = read_cwru_mat(path("raw_dir") / e.filename, e.file_number).signal
        main = analyze(sig, e.fs, e.rpm, e.load_hp, run_whatif=False).diagnosis
        assert edge["ml_fault"] == main["ml_fault"], e.name
        assert edge["fault"] == main["fault"], e.name
        assert edge["decision_source"] == main["decision_source"], e.name
        assert abs(edge["confidence"] - main["confidence"]) < 0.02, e.name


@needs_bundle
def test_edge_bundle_has_no_heavy_imports():
    import re

    src = (EDGE / "edge_infer.py").read_text(encoding="utf-8") + (EDGE / "edge_features.py").read_text(encoding="utf-8")
    # module-level imports only (indented imports inside functions are lazy and never run on the device)
    top_imports = [l for l in src.splitlines() if re.match(r"^(import|from)\s", l)]
    modules = {re.match(r"^(?:import|from)\s+([\w.]+)", l).group(1).split(".")[0] for l in top_imports}
    assert modules <= {"__future__", "argparse", "json", "re", "sys", "fractions", "pathlib",
                       "numpy", "onnxruntime", "scipy", "edge_features"}, modules

"""Step 4 (Edge AI): size-vs-accuracy study, ONNX export + parity check, latency/size/memory benchmark,
and the standalone edge bundle in edge/.

All timings are MEASURED ON THE DEVELOPMENT LAPTOP (Windows, CPU), NOT ON EDGE HARDWARE.

Outputs
  models/fault_rf_edge.joblib/.onnx, severity_rf_edge.joblib/.onnx, fault_rf.onnx
  edge/  (edge_infer.py + edge_features.py + edge_config.json + *.onnx)
  reports/metrics/edge_size_study.csv, edge_benchmark.json
  reports/figures/edge_size_vs_f1.png, edge_latency.png
"""
import _bootstrap  # noqa: F401

import io
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pdm.config import load_config, path
from pdm.data.cwru_catalog import CATALOG, CLASSES
from pdm.data.loader import load_cwru_recording
from pdm.edge.benchmark import LABEL, file_size_kb, machine_info, peak_python_memory_mb, process_memory_mb, time_fn
from pdm.edge.export import onnx_predict, onnx_session, parity_check, to_onnx, write_edge_bundle
from pdm.ml.evaluate import lolo_evaluate, size_fold_column
from pdm.ml.train import all_feature_cols, fit_fault_model, fit_severity_model, load_features, load_model, save_model
from pdm.signal.features import extract_features, feature_names, params_from_config
from pdm.signal.windowing import window_signal

TREES = [10, 25, 50, 100, 200]
DEPTHS = [8, 12, 16, None]
SIZE_LABELS = ["0.007", "0.014", "0.021", "0.028"]


def main() -> None:
    df = load_features()
    df["size_fold"] = size_fold_column(df)
    cols = all_feature_cols()
    sev_cols = all_feature_cols("severity_rf")
    models_dir, metrics_dir, fig_dir = path("models_dir"), path("metrics_dir"), path("figures_dir")
    tmp_dir = models_dir / "_size_study"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- 1. size vs accuracy
    rows = []
    for n in TREES:
        for d in DEPTHS:
            ov = {"n_estimators": n, "max_depth": d}
            lolo = lolo_evaluate(df, cols, "fault", CLASSES, rf_overrides=ov, keep_oof=False)
            loso = lolo_evaluate(df, cols, "fault", CLASSES, rf_overrides=ov, keep_oof=False, fold_col="size_fold")
            m = fit_fault_model(df, cols, **ov)
            buf = io.BytesIO(); joblib.dump(m, buf)
            onnx_p = to_onnx(m, len(cols), tmp_dir / f"rf_{n}_{d}.onnx")
            rows.append({"n_estimators": n, "max_depth": "none" if d is None else d,
                         "total_nodes": int(sum(t.tree_.node_count for t in m.estimators_)),
                         "joblib_kb": buf.getbuffer().nbytes / 1024, "onnx_kb": file_size_kb(onnx_p),
                         "lolo_macro_f1": lolo["window"]["macro_f1"]["mean"],
                         "unseen_bearing_macro_f1": loso["window"]["macro_f1"]["mean"]})
            print(f"trees={n:3d} depth={str(d):4s} onnx={rows[-1]['onnx_kb']:8.1f} KB  "
                  f"LOLO F1={rows[-1]['lolo_macro_f1']:.4f}  unseen={rows[-1]['unseen_bearing_macro_f1']:.3f}")
    study = pd.DataFrame(rows)
    study.to_csv(metrics_dir / "edge_size_study.csv", index=False)
    for f in tmp_dir.glob("*.onnx"):
        f.unlink()
    tmp_dir.rmdir()

    best = study.lolo_macro_f1.max()
    ok = study[study.lolo_macro_f1 >= best - 0.01].sort_values("onnx_kb")
    chosen = ok.iloc[0]
    ch_depth = None if chosen.max_depth == "none" else int(chosen.max_depth)
    ch = {"n_estimators": int(chosen.n_estimators), "max_depth": ch_depth}
    print(f"\nChosen edge model: {ch} ({chosen.onnx_kb:.1f} KB ONNX, LOLO F1 {chosen.lolo_macro_f1:.4f}; "
          f"best {best:.4f})")

    # ------------------------------------------------------------- 2. train + export
    edge_fault = fit_fault_model(df, cols, **ch)
    edge_sev = fit_severity_model(df, sev_cols, **ch)
    save_model(edge_fault, "fault_rf_edge", cols, {"chosen_by": "smallest ONNX within 1% LOLO macro-F1 of best",
                                                   "lolo_macro_f1": float(chosen.lolo_macro_f1)})
    save_model(edge_sev, "severity_rf_edge", sev_cols, {"params": ch})
    full = load_model("fault_rf")["model"]
    p_full = to_onnx(full, len(cols), models_dir / "fault_rf.onnx")
    p_edge = to_onnx(edge_fault, len(cols), models_dir / "fault_rf_edge.onnx")
    p_sev = to_onnx(edge_sev, len(sev_cols), models_dir / "severity_rf_edge.onnx")

    X = df[cols].to_numpy()
    Xs = df[df.fault != "Normal"][sev_cols].to_numpy()
    s_full, s_edge, s_sev = onnx_session(p_full), onnx_session(p_edge), onnx_session(p_sev)
    parity = {"fault_rf (full)": parity_check(full, s_full, X),
              "fault_rf_edge": parity_check(edge_fault, s_edge, X),
              "severity_rf_edge": parity_check(edge_sev, s_sev, Xs)}
    for k, v in parity.items():
        print(f"parity {k:18s}: label agreement {v['label_agreement']:.5f} ({v['label_mismatches']} of {v['n']} differ), "
              f"max |prob diff| {v['max_abs_prob_diff']:.2e}")
    edge_vs_full = float(np.mean(edge_fault.predict(X) == full.predict(X)))

    write_edge_bundle({"fault": p_edge, "severity": p_sev, "fault_full": p_full}, cols, sev_cols,
                      {"fault": [str(c) for c in edge_fault.classes_],
                       "severity": [str(c) for c in edge_sev.classes_]})

    # ------------------------------------------------------------- 3. benchmark
    params = params_from_config()
    fs = load_config()["data"]["target_fs"]
    entry = next(e for e in CATALOG if e.name == "OR014@6_2")
    sig, rpm, _ = load_cwru_recording(entry)
    win = window_signal(sig)
    w1 = win[:1]
    names = feature_names(params)
    idx = [names.index(c) for c in cols]
    feats = lambda w: extract_features(w, fs, rpm, params)[:, idx]  # model columns only  # noqa: E731
    x1 = feats(w1)
    x1f = x1.astype(np.float32)
    in_full, in_edge = s_full.get_inputs()[0].name, s_edge.get_inputs()[0].name
    full_1 = load_model("fault_rf")["model"]
    full_1.set_params(n_jobs=1); edge_fault.set_params(n_jobs=1)

    lat = {
        "feature_extraction_per_window": time_fn(lambda: feats(w1)),
        "inference_joblib_full_per_window": time_fn(lambda: full_1.predict_proba(x1), repeats=300),
        "inference_joblib_edge_per_window": time_fn(lambda: edge_fault.predict_proba(x1), repeats=300),
        "inference_onnx_full_per_window": time_fn(lambda: s_full.run(None, {in_full: x1f})),
        "inference_onnx_edge_per_window": time_fn(lambda: s_edge.run(None, {in_edge: x1f})),
        "end_to_end_onnx_edge_per_window": time_fn(
            lambda: s_edge.run(None, {in_edge: feats(w1).astype(np.float32)})),
        "end_to_end_onnx_edge_full_recording": time_fn(
            lambda: onnx_predict(s_edge, feats(win)), repeats=50, warmup=3),
    }
    mem = {
        "peak_python_alloc_features_plus_onnx_per_window_mb": peak_python_memory_mb(
            lambda: s_edge.run(None, {in_edge: feats(w1).astype(np.float32)})),
        "peak_python_alloc_full_recording_mb": peak_python_memory_mb(
            lambda: onnx_predict(s_edge, feats(win))),
        "process_rss_mb_after_loading_everything": process_memory_mb(),
    }
    sizes = {p.name: round(file_size_kb(p), 1) for p in sorted(models_dir.glob("*")) if p.suffix in (".joblib", ".onnx")}
    bench = {"label": LABEL, "machine": machine_info(), "chosen_edge_model": ch,
             "chosen_edge_lolo_macro_f1": float(chosen.lolo_macro_f1),
             "chosen_edge_unseen_bearing_macro_f1": float(chosen.unseen_bearing_macro_f1),
             "edge_vs_full_window_agreement": edge_vs_full, "parity": parity,
             "latency": lat, "memory": mem, "file_sizes_kb": sizes,
             "benchmark_recording": {"recording_id": entry.recording_id, "windows": int(len(win)),
                                     "seconds_of_signal": round(len(sig) / fs, 2)}}
    (metrics_dir / "edge_benchmark.json").write_text(json.dumps(bench, indent=2), encoding="utf-8")

    # ------------------------------------------------------------- 4. plots
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for d, g in study.groupby("max_depth"):
        ax.plot(g.onnx_kb, g.lolo_macro_f1, "o-", label=f"max_depth={d}")
    ax.scatter([chosen.onnx_kb], [chosen.lolo_macro_f1], s=180, facecolors="none", edgecolors="red", lw=2,
               label="chosen", zorder=5)
    ax.set_xscale("log"); ax.set_xlabel("ONNX model size (KB, log scale)"); ax.set_ylabel("LOLO macro-F1")
    ax.set_title("Edge AI: model size vs accuracy (labels: number of trees along each line)", fontsize=9)
    for _, r in study[study.max_depth == "none"].iterrows():
        ax.annotate(int(r.n_estimators), (r.onnx_kb, r.lolo_macro_f1), fontsize=7, xytext=(3, -10),
                    textcoords="offset points")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(fig_dir / "edge_size_vs_f1.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    keys = [k for k in lat if k.endswith("per_window")]
    ax.barh([k.replace("_per_window", "").replace("_", " ") for k in keys], [lat[k]["median_ms"] for k in keys],
            xerr=[[0] * len(keys), [lat[k]["p95_ms"] - lat[k]["median_ms"] for k in keys]], color="#2a9d8f")
    ax.set_xlabel("ms per 2048-sample window (median, bar to p95)")
    ax.set_title(LABEL, fontsize=8); fig.tight_layout(); fig.savefig(fig_dir / "edge_latency.png", dpi=130); plt.close(fig)

    print("\nLatency (median / p95 ms):")
    for k, v in lat.items():
        print(f"  {k:40s} {v['median_ms']:8.3f} / {v['p95_ms']:8.3f}")
    print("Memory:", {k: round(v, 2) for k, v in mem.items()})
    print("File sizes (KB):", sizes)
    print(f"Compact vs full model window agreement: {edge_vs_full:.4f}")
    print(LABEL)


if __name__ == "__main__":
    main()

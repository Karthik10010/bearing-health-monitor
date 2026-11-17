"""Step 3: train + evaluate the fault-type RF and the fault-size (severity) RF.

Outputs
  models/fault_rf.joblib, models/severity_rf.joblib (+ .json model cards)
  reports/metrics/fault_metrics.json, severity_metrics.json, feature_groups.json, class_feature_stats.json
  reports/metrics/fault_oof.csv, severity_oof.csv   (out-of-fold predictions; used by the RL simulator)
  reports/figures/*.png
"""
import _bootstrap  # noqa: F401

import json

import numpy as np

from pdm.config import path
from pdm.data.cwru_catalog import CLASSES
from pdm.data.cwru_catalog import HOLDOUT as HOLDOUT_FILES
from pdm.ml.evaluate import (feature_group_comparison, lolo_evaluate, naive_split_evaluate, normalise_rows,
                             size_fold_column)
from pdm.ml.plots import bar_plot, confusion_plot, importance_plot
from pdm.ml.train import (all_feature_cols, computed_feature_cols, faulty_only, fit_fault_model, fit_severity_model,
                          load_features, save_model)
from pdm.physics import knock_evidence
from pdm.signal.features import extract_features, feature_groups, feature_names, params_from_config
from pdm.signal.windowing import window_signal

# Extra CWRU outer-race recordings (@3 and @12 o'clock) that are NEVER used for training or for any
# design decision - an independent final check. Downloaded into data/unseen/ when available.
HOLDOUT = {e.file_number: e.recording_id for e in HOLDOUT_FILES}


def physics_call(g) -> str | None:
    med = {k: float(g[k].median()) for k in ("e_bpfo_1", "e_bpfi_1", "e_bsf_1", "e_bsf_2")}
    return knock_evidence(med).call


def hybrid_recording_eval(df, loso) -> dict:
    """Recording-level accuracy on UNSEEN bearings for: ML only, physics only, and ML + physics
    (physics decides when it finds a clear knock signature, otherwise ML). The physics rule has no
    trained parameters, so applying it inside the leave-one-size-out folds is fair."""
    ml = {r["recording_id"]: r["pred"] for f in loso["recording"]["folds"] for r in f["recordings"]}
    rows = []
    for rid, g in df.groupby("recording_id"):
        truth, phys = g.fault.iloc[0], physics_call(g)
        rows.append((truth, ml[rid], phys, phys or ml[rid]))
    n = len(rows)
    called = [r for r in rows if r[2]]
    return {"recordings": n,
            "ml_only_accuracy": round(sum(t == m for t, m, _, _ in rows) / n, 4),
            "hybrid_accuracy": round(sum(t == h for t, _, _, h in rows) / n, 4),
            "physics_calls": len(called), "physics_calls_correct": sum(t == p for t, _, p, _ in called),
            "faulty_detected_as_faulty_hybrid": round(
                sum((h != "Normal") for t, _, _, h in rows if t != "Normal") / sum(t != "Normal" for t, *_ in rows), 4)}


def holdout_check(model, cols) -> dict | None:
    raw = path("unseen_dir")
    if not all((raw / f"{n}.mat").exists() for n in HOLDOUT):
        return None
    from pdm.data.loader import read_cwru_mat
    params, names = params_from_config(), feature_names(params_from_config())
    idx = [names.index(c) for c in cols]
    out = []
    for num, rid in HOLDOUT.items():
        ms = read_cwru_mat(raw / f"{num}.mat", num)
        X = extract_features(window_signal(ms.signal), 12000, ms.rpm, params)
        p = model.predict_proba(X[:, idx]).mean(axis=0)
        ml = str(model.classes_[p.argmax()])
        med = {k: float(np.median(X[:, names.index(k)])) for k in ("e_bpfo_1", "e_bpfi_1", "e_bsf_1", "e_bsf_2")}
        phys = knock_evidence(med).call
        out.append({"file": f"{num}.mat", "recording": rid, "truth": "Outer race", "ml": ml,
                    "ml_confidence": round(float(p.max()), 3), "physics": phys, "hybrid": phys or ml})
    return {"note": "CWRU outer race @3/@12 o'clock: never used for training or design decisions",
            "results": out, "ml_correct": sum(r["ml"] == r["truth"] for r in out),
            "hybrid_correct": sum(r["hybrid"] == r["truth"] for r in out), "n": len(out)}

SIZE_LABELS = ["0.007", "0.014", "0.021", "0.028"]


def strip(res: dict) -> dict:
    return {k: v for k, v in res.items() if k != "oof"}


def main() -> None:
    df = load_features()
    cols = all_feature_cols()
    metrics_dir, fig_dir = path("metrics_dir"), path("figures_dir")
    metrics_dir.mkdir(parents=True, exist_ok=True); fig_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- fault type
    print("Fault RF: leave-one-load-out ...")
    fault = lolo_evaluate(df, cols, "fault", CLASSES)
    naive = naive_split_evaluate(df, cols, "fault", CLASSES)
    print("Fault RF: feature-group comparison ...")
    groups = feature_group_comparison(df, feature_groups(cols), "fault", CLASSES)
    fault["oof"].to_csv(metrics_dir / "fault_oof.csv", index=False)

    # Harder, more honest test: CWRU records the SAME damaged bearing at all 4 loads, so LOLO can
    # partly "recognise the bearing". Leave-one-fault-size-out hides whole bearings (all loads).
    print("Fault RF: leave-one-fault-size-out (unseen bearings) ...")
    df["size_fold"] = size_fold_column(df)
    loso = lolo_evaluate(df, cols, "fault", CLASSES, fold_col="size_fold")
    loso["oof"].to_csv(metrics_dir / "fault_oof_unseen_bearing.csv", index=False)
    groups_loso = feature_group_comparison(df, feature_groups(cols), "fault", CLASSES, fold_col="size_fold")
    hybrid = hybrid_recording_eval(df, loso)

    final = fit_fault_model(df, cols)
    importances = dict(sorted(zip(cols, final.feature_importances_.round(5).tolist()), key=lambda kv: -kv[1]))
    fault_out = {
        **strip(fault),
        "window_confusion_normalised": normalise_rows(fault["window"]["confusion"]),
        "recording_confusion_normalised": normalise_rows(fault["recording"]["confusion"]),
        "naive_random_split": naive,
        "leave_one_size_out": {**strip(loso),
                               "window_confusion_normalised": normalise_rows(loso["window"]["confusion"]),
                               "recording_confusion_normalised": normalise_rows(loso["recording"]["confusion"]),
                               "note": "fold k = all fault recordings of size k + Normal at load k; "
                                       "tests on physical bearings never seen in training"},
        "hybrid_unseen_bearings": hybrid,
        "independent_holdout": holdout_check(final, cols),
        "model_features": cols,
        "feature_importance": importances,
        "n_windows": int(len(df)), "n_recordings": int(df.recording_id.nunique()),
    }
    (metrics_dir / "fault_metrics.json").write_text(json.dumps(fault_out, indent=2), encoding="utf-8")
    (metrics_dir / "feature_groups.json").write_text(
        json.dumps({"leave_one_load_out": groups, "leave_one_size_out": groups_loso}, indent=2), encoding="utf-8")
    save_model(final, "fault_rf", cols, {"target": "fault",
                                         "lolo_window_macro_f1": fault["window"]["macro_f1"]["mean"],
                                         "lolo_recording_accuracy": fault["recording"]["accuracy"]["mean"]})

    confusion_plot(fault["window"]["confusion"], CLASSES, "Fault RF - LOLO, window level (all 4 folds)",
                   fig_dir / "fault_confusion_window.png")
    confusion_plot(fault["recording"]["confusion"], CLASSES, "Fault RF - LOLO, recording level",
                   fig_dir / "fault_confusion_recording.png")
    bar_plot({k: v["window_macro_f1"]["mean"] for k, v in groups.items()},
             {k: v["window_macro_f1"]["std"] for k, v in groups.items()},
             "Fault RF - LOLO macro-F1 by feature group (mean +/- std over 4 loads)", "macro F1",
             fig_dir / "feature_group_comparison.png")
    bar_plot({k: v["window_macro_f1"]["mean"] for k, v in groups_loso.items()},
             {k: v["window_macro_f1"]["std"] for k, v in groups_loso.items()},
             "Fault RF - unseen bearings (leave-one-size-out) macro-F1 by feature group", "macro F1",
             fig_dir / "feature_group_comparison_unseen_bearing.png")
    confusion_plot(loso["window"]["confusion"], CLASSES, "Fault RF - unseen bearings (leave-one-size-out), window level",
                   fig_dir / "fault_confusion_unseen_bearing.png")
    split_names = ["Naive random split\n(leaky)", "Leave-one-load-out", "Leave-one-size-out\n(unseen bearings)"]
    bar_plot(dict(zip(split_names, [naive["macro_f1"], fault["window"]["macro_f1"]["mean"],
                                    loso["window"]["macro_f1"]["mean"]])),
             dict(zip(split_names, [0.0, fault["window"]["macro_f1"]["std"], loso["window"]["macro_f1"]["std"]])),
             "How strict the exam is changes the score: window-level macro-F1", "macro F1",
             fig_dir / "leakage_demo.png")
    importance_plot(cols, final.feature_importances_, fig_dir / "feature_importance.png")

    # ------------------------------------------------------------- severity
    print("Severity RF: leave-one-load-out ...")
    fdf = faulty_only(df)
    sev_cols = all_feature_cols("severity_rf")
    sev = lolo_evaluate(fdf, sev_cols, "size_label", SIZE_LABELS, kind="severity_rf")
    sev["oof"].to_csv(metrics_dir / "severity_oof.csv", index=False)
    sev_naive = naive_split_evaluate(fdf, sev_cols, "size_label", SIZE_LABELS, kind="severity_rf")
    # Harder test for size: train on two fault types, test on the third (unseen bearings again).
    print("Severity RF: leave-one-fault-type-out ...")
    sev_lofo = lolo_evaluate(fdf, sev_cols, "size_label", SIZE_LABELS, kind="severity_rf", fold_col="fault")
    sev_lofo.pop("oof").to_csv(metrics_dir / "severity_oof_unseen_type.csv", index=False)
    sev_final = fit_severity_model(df, sev_cols)
    sev_out = {**strip(sev),
               "window_confusion_normalised": normalise_rows(sev["window"]["confusion"]),
               "recording_confusion_normalised": normalise_rows(sev["recording"]["confusion"]),
               "naive_random_split": sev_naive,
               "leave_one_fault_type_out": {**sev_lofo,
                                            "window_confusion_normalised": normalise_rows(sev_lofo["window"]["confusion"]),
                                            "note": "train on 2 fault types, test on the 3rd"}}
    confusion_plot(sev_lofo["window"]["confusion"], [s + '"' for s in SIZE_LABELS],
                   "Severity RF - unseen fault type, window level", fig_dir / "severity_confusion_unseen_type.png")
    (metrics_dir / "severity_metrics.json").write_text(json.dumps(sev_out, indent=2), encoding="utf-8")
    save_model(sev_final, "severity_rf", sev_cols, {"target": "size_in (inches, as class label)",
                                               "trained_on": "faulty windows only",
                                               "lolo_window_macro_f1": sev["window"]["macro_f1"]["mean"]})
    confusion_plot(sev["window"]["confusion"], [s + '"' for s in SIZE_LABELS],
                   "Severity RF - LOLO, window level", fig_dir / "severity_confusion_window.png")
    confusion_plot(sev["recording"]["confusion"], [s + '"' for s in SIZE_LABELS],
                   "Severity RF - LOLO, recording level", fig_dir / "severity_confusion_recording.png")

    # ------------------------------------------------------------- training-data fingerprints
    # Lets analyze() warn when it is given a recording the models were trained on (not an independent test).
    from pdm.data.cwru_catalog import CATALOG
    from pdm.data.loader import read_cwru_mat
    from pdm.pipeline import signal_fingerprint
    fps = {e.recording_id: signal_fingerprint(read_cwru_mat(path("raw_dir") / e.filename, e.file_number).signal)
           for e in CATALOG}
    (path("models_dir") / "training_fingerprints.json").write_text(json.dumps(fps, indent=1), encoding="utf-8")

    # ------------------------------------------------------------- reference stats for explanations
    stats = {}
    every = computed_feature_cols()
    for c in CLASSES:
        d = df[df.fault == c][every]
        stats[c] = {k: {"median": float(np.median(d[k])), "p25": float(np.percentile(d[k], 25)),
                        "p75": float(np.percentile(d[k], 75))} for k in every}
    (metrics_dir / "class_feature_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # ------------------------------------------------------------- summary
    w, r = fault["window"], fault["recording"]
    print(f"\nFAULT  LOLO window macro-F1 {w['macro_f1']['mean']:.3f} +/- {w['macro_f1']['std']:.3f} "
          f"(per load {w['macro_f1']['per_fold']}) | window acc {w['accuracy']['mean']:.3f}")
    print(f"FAULT  LOLO recording accuracy {r['accuracy']['mean']:.3f} +/- {r['accuracy']['std']:.3f}")
    print(f"FAULT  naive random split macro-F1 {naive['macro_f1']:.3f} (leaky)")
    print(f"HYBRID (ML + physics) unseen-bearing recording accuracy {hybrid['hybrid_accuracy']:.3f} "
          f"vs ML only {hybrid['ml_only_accuracy']:.3f}; physics spoke on {hybrid['physics_calls']} recordings, "
          f"{hybrid['physics_calls_correct']} correct")
    ho = fault_out["independent_holdout"]
    if ho:
        print(f"INDEPENDENT HOLD-OUT (4 never-used outer-race recordings): ML {ho['ml_correct']}/4, "
              f"ML + physics {ho['hybrid_correct']}/4 -> {[(r['file'], r['ml'], r['physics']) for r in ho['results']]}")
    lw, lr = loso["window"], loso["recording"]
    print(f"FAULT  UNSEEN-BEARING window macro-F1 {lw['macro_f1']['mean']:.3f} +/- {lw['macro_f1']['std']:.3f} "
          f"(per fold {lw['macro_f1']['per_fold']}) | recording acc {lr['accuracy']['mean']:.3f}")
    for k in groups:
        a, b = groups[k]["window_macro_f1"], groups_loso[k]["window_macro_f1"]
        print(f"  group {k:9s} ({groups[k]['n_features']:2d} feats): LOLO {a['mean']:.3f} +/- {a['std']:.3f} | "
              f"unseen-bearing {b['mean']:.3f} +/- {b['std']:.3f}")
    sw, sr = sev["window"], sev["recording"]
    print(f"SEVERITY LOLO window macro-F1 {sw['macro_f1']['mean']:.3f} +/- {sw['macro_f1']['std']:.3f} | "
          f"recording acc {sr['accuracy']['mean']:.3f} | naive {sev_naive['macro_f1']:.3f}")
    print(f"SEVERITY unseen fault type: window macro-F1 {sev_lofo['window']['macro_f1']['mean']:.3f} "
          f"(per type {sev_lofo['window']['macro_f1']['per_fold']}) | window acc "
          f"{sev_lofo['window']['accuracy']['mean']:.3f}")
    print("Fault confusion, UNSEEN BEARINGS (window, rows=true):", CLASSES)
    for c, row in zip(CLASSES, normalise_rows(loso["window"]["confusion"])):
        print(f"  {c:11s}", row)
    print("Unseen-bearing recordings misclassified:",
          [r for f in lr["folds"] for r in f["recordings"] if r["true"] != r["pred"]])
    print("Fault confusion (window, rows=true):", CLASSES)
    for c, row in zip(CLASSES, fault_out["window_confusion_normalised"]):
        print(f"  {c:11s}", row)
    print("Severity confusion (window, rows=true):", SIZE_LABELS)
    for c, row in zip(SIZE_LABELS, sev_out["window_confusion_normalised"]):
        print(f"  {c:6s}", row)


if __name__ == "__main__":
    main()

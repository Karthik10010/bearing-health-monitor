import io
import zipfile
import pandas as pd
import plotly.express as px
import streamlit as st

from components.common import ROOT, metrics
from pdm.config import path

st.title("Edge AI")
b = metrics("edge_benchmark.json")
if not b:
    st.info("Run scripts/04_edge_benchmark.py to create the edge model and benchmark.")
    st.stop()

st.warning(f"**{b['label']}** Machine: {b['machine']['processor'] or b['machine']['platform']}, "
           f"{b['machine']['logical_cpus']} logical CPUs, {b['machine']['ram_gb']} GB RAM.")
ch = b["chosen_edge_model"]
c = st.columns(4)
c[0].metric("Edge model", f"{ch['n_estimators']} trees", f"max depth {ch['max_depth'] or 'unlimited'}",
            delta_color="off", delta_arrow="off")
c[1].metric("ONNX size", f"{b['file_sizes_kb']['fault_rf_edge.onnx']:.0f} KB",
            f"full model {b['file_sizes_kb']['fault_rf.onnx']:.0f} KB", delta_color="off", delta_arrow="off")
c[2].metric("End-to-end per window", f"{b['latency']['end_to_end_onnx_edge_per_window']['median_ms']:.1f} ms",
            f"p95 {b['latency']['end_to_end_onnx_edge_per_window']['p95_ms']:.1f} ms", delta_color="off", delta_arrow="off")
c[3].metric("LOLO macro-F1 (edge)", f"{b['chosen_edge_lolo_macro_f1']:.3f}",
            f"unseen bearings {b['chosen_edge_unseen_bearing_macro_f1']:.3f}", delta_color="off", delta_arrow="off")

st.subheader("Model size vs. accuracy")
study = metrics("edge_size_study.csv")
if study:
    df = pd.read_csv(study)
    df["max_depth"] = df["max_depth"].astype(str)
    fig = px.line(df.sort_values("onnx_kb"), x="onnx_kb", y="lolo_macro_f1", color="max_depth", markers=True,
                  log_x=True, hover_data=["n_estimators", "unseen_bearing_macro_f1"],
                  labels={"onnx_kb": "ONNX size (KB, log)", "lolo_macro_f1": "LOLO macro-F1"}, height=360)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("The edge model is the smallest one within 1% macro-F1 of the best. Hover to see the unseen-bearing "
               "score too.")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Latency (ms)")
    st.dataframe(pd.DataFrame(b["latency"]).T[["median_ms", "p95_ms", "repeats"]].round(3), use_container_width=True)
    st.caption(f"Full recording = {b['benchmark_recording']['windows']} windows "
               f"({b['benchmark_recording']['seconds_of_signal']} s of signal).")
with c2:
    st.subheader("Memory & files")
    st.dataframe(pd.Series(b["memory"]).round(2).rename("MB"), use_container_width=True)
    st.dataframe(pd.Series(b["file_sizes_kb"]).rename("KB"), use_container_width=True)
    st.subheader("ONNX vs. scikit-learn parity")
    st.dataframe(pd.DataFrame(b["parity"]).T, use_container_width=True)

st.subheader("Downloads")
edge_dir = ROOT / "edge"
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for f in ("edge_infer.py", "edge_features.py", "edge_config.json", "fault.onnx", "severity.onnx"):
        if (edge_dir / f).exists():
            z.write(edge_dir / f, f"edge/{f}")
d = st.columns(3)
d[0].download_button("Edge bundle (.zip)", buf.getvalue(), "bearing_edge_bundle.zip", "application/zip",
                     use_container_width=True)
for col, f in zip(d[1:], ("fault_rf_edge.onnx", "fault_rf_edge.joblib")):
    p = path("models_dir") / f
    if p.exists():
        col.download_button(f, p.read_bytes(), f, use_container_width=True)
st.code("pip install numpy scipy onnxruntime\npython edge/edge_infer.py my_signal.npy --fs 12000 --rpm 1750",
        language="bash")

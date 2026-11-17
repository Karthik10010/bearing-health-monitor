import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.common import get_report

st.title("Signal Analysis")
rep = get_report()
if rep is not None:
    a = rep.arrays
    fs = a["fs_used"]
    x = a["signal_12k"]

    with st.container(border=True):
        st.markdown("**Raw vibration waveform**")
        st.caption(f"Resampled to {fs} Hz (original {rep.input['fs_original_hz']:g} Hz). Drag to zoom, double-click to reset.")
        step = max(1, len(x) // 20000)
        t = np.arange(0, len(x), step) / fs
        fig = go.Figure(go.Scattergl(x=t, y=x[::step], mode="lines", line=dict(width=0.8)))
        fig.update_layout(height=260, xaxis_title="time (s)", yaxis_title="acceleration", margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.markdown("**FFT spectrum** — which frequencies the vibration contains")
        fig = go.Figure(go.Scattergl(x=a["fft_freq"], y=a["fft_mag"], mode="lines", line=dict(width=0.8)))
        fig.update_layout(height=330, xaxis_title="frequency (Hz)", yaxis_title="amplitude", yaxis_type="log",
                          margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2, st.container(border=True):
        st.markdown("**Envelope spectrum** — repeating knocks show up as peaks")
        fig = go.Figure(go.Scatter(x=a["env_freq"], y=a["env_mag"], mode="lines", line=dict(width=1), name="envelope"))
        colors = {"bpfo": "#d1495b", "bpfi": "#1f9d7a", "bsf": "#e0a43a"}
        names = {"bpfo": "BPFO (outer race)", "bpfi": "BPFI (inner race)", "bsf": "BSF (ball)"}
        for k, col in colors.items():
            f0 = rep.defect_frequencies_hz[k]
            for hh in (1, 2, 3):
                fig.add_vline(x=hh * f0, line=dict(color=col, dash="dash", width=1))
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=col, dash="dash"),
                                     name=f"{names[k]} {f0:.1f} Hz ×1,2,3"))
        fig.update_layout(height=330, xaxis_title="frequency (Hz)", yaxis_title="amplitude", margin=dict(t=10, b=10),
                          legend=dict(font=dict(size=10), orientation="h", y=-0.3))
        st.plotly_chart(fig, use_container_width=True)

    with st.container(border=True):
        st.markdown("**Defect-frequency peaks** (× background level)")
        st.caption(rep.envelope_peaks["meaning"])
        rows = [{"defect": k.upper(), "frequency (Hz)": v["frequency_hz"], "1x": v["peak_ratio_h1"],
                 "2x": v.get("peak_ratio_h2"), "3x": v.get("peak_ratio_h3"), "healthy median (1x)": v.get("healthy_median_h1")}
                for k, v in rep.envelope_peaks.items() if isinstance(v, dict)]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with st.container(border=True):
        st.markdown("**All 37 measurements vs. healthy bearings**")
        st.caption("Median over all windows of this recording, next to the median of healthy bearings and of the "
                   "diagnosed class in the CWRU training data. Absolute-size measurements (mean, std, RMS, min, max, "
                   "peak-to-peak) are shown for context but not used by the fault-type model.")
        st.dataframe(pd.DataFrame(a["feature_table"]), hide_index=True, use_container_width=True, height=420)

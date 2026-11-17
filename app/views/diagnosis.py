import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from components.common import badge, get_report, metrics
from pdm.config import load_config, path

st.title("Fault Diagnosis")
rep = get_report()
if rep is not None:
    d, h = rep.diagnosis, rep.health
    ke = d["knock_evidence"]
    rules = load_config()["physics"]

    with st.container(border=True):
        st.markdown(f"### {d['fault']}")
        st.html(badge(h["status"], h["low_confidence"]))
        st.caption(f"Decided by **{d['decision_source']}** · {h['label']}: " + "; ".join(h["reasons"]))

    st.markdown("#### Two independent opinions")
    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.markdown(f"**① ML model (Random Forest)** → {d['ml_fault']} ({d['ml_confidence']:.0%})")
        p = d["probabilities"]
        fig = px.bar(x=list(p.values()), y=list(p), orientation="h", labels={"x": "share of tree votes", "y": ""},
                     range_x=[0, 1], height=230, text=[f"{v:.0%}" for v in p.values()])
        fig.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"{d['window_agreement']:.0%} of the {d['n_windows']} windows agree with the final diagnosis.")
    with c2, st.container(border=True):
        verdict = ke["call"] or "no clear signature"
        st.markdown(f"**② Physics knock check** → {verdict}")
        pk = ke["peak_ratios"]
        fig = go.Figure(go.Bar(x=list(pk.values()), y=list(pk), orientation="h",
                               text=[f"{v:.1f}×" for v in pk.values()], marker_color="#e0a43a"))
        fig.add_vline(x=rules["min_peak_ratio"], line=dict(dash="dash", color="#d1495b"),
                      annotation_text=f"clear = {rules['min_peak_ratio']:g}×", annotation_position="top")
        fig.update_layout(height=230, margin=dict(t=25, b=10), xaxis_title="knock peak (× background)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Rule: {ke['rule']}. Strongest: {ke['strongest']} ({ke['dominance']}× the next). "
                   "On CWRU it made 19 calls, all correct; it stays silent when no knock is clear.")
    if not d["methods_agree"]:
        st.warning(f"The two opinions disagree. The clear knock signature ({ke['call']}) is used, and the result is "
                   "flagged low-confidence — an inspection is advisable.", icon="⚖️")

    c3, c4 = st.columns(2)
    with c3, st.container(border=True):
        st.markdown("**Estimated fault size** (severity model)")
        if d["size_in"]:
            st.metric("size", f'{d["size_in"]:.3f}"', f"{d['size_confidence']:.0%} of votes", delta_color="off",
                      delta_arrow="off")
            sp = d["size_probabilities"]
            fig = px.bar(x=[f'{k}"' for k in sp], y=list(sp.values()), labels={"x": "", "y": "probability"}, height=200)
            fig.update_layout(margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No fault diagnosed — no size estimate.")
        st.caption("Indicative only: size estimates did not transfer to unseen fault types in testing.")
    with c4, st.container(border=True):
        st.markdown("**Per-window ML predictions**")
        wp = rep.arrays["window_predictions"]
        fig = go.Figure(go.Scatter(x=list(range(len(wp))), y=wp, mode="markers", marker=dict(size=6)))
        fig.update_layout(height=250, margin=dict(t=10, b=10), xaxis_title="window #",
                          yaxis=dict(categoryorder="array", categoryarray=["Normal", "Inner race", "Outer race", "Ball"]))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### How good is it? (evaluation on CWRU)")
    ev = rep.model_evaluation
    fm = metrics("fault_metrics.json") or {}
    hy = fm.get("hybrid_unseen_bearings", {})
    ho = fm.get("independent_holdout")
    m = st.columns(4)
    m[0].metric("Seen bearings (other loads)", f"{ev['fault_type_seen_bearings_lolo']['window_macro_f1']:.3f}",
                "macro-F1, ML", delta_color="off", delta_arrow="off")
    m[1].metric("Unseen bearings — ML only", f"{hy.get('ml_only_accuracy', 0):.0%}", "recordings correct",
                delta_color="off", delta_arrow="off")
    m[2].metric("Unseen bearings — ML + physics", f"{hy.get('hybrid_accuracy', 0):.0%}", "recordings correct",
                delta_color="off", delta_arrow="off")
    if ho:
        m[3].metric("Never-used recordings", f"{ho['hybrid_correct']}/{ho['n']}", f"ML alone {ho['ml_correct']}/{ho['n']}",
                    delta_color="off", delta_arrow="off")
    st.info("**Why several numbers?** CWRU recorded each damaged bearing at all four loads, so a test on another load "
            "still shows the model the *same physical bearing*. Testing on bearings it has never seen is the honest "
            "number for a new machine. Healthy vs damaged stays reliable "
            f"({ev['fault_type_unseen_bearings']['normal_windows_correct']:.0%} of healthy windows correct); the fault "
            "*type* improves a lot when the physics knock check is combined with the ML model.", icon="ℹ️")
    figs = [("Confusion matrix — seen bearings (LOLO)", "fault_confusion_window.png"),
            ("Confusion matrix — unseen bearings (ML only)", "fault_confusion_unseen_bearing.png"),
            ("Feature groups (LOLO)", "feature_group_comparison.png"),
            ("Feature groups (unseen bearings)", "feature_group_comparison_unseen_bearing.png"),
            ("Split strictness vs. score", "leakage_demo.png"), ("Feature importance", "feature_importance.png"),
            ("Severity — LOLO", "severity_confusion_window.png"),
            ("Severity — unseen fault type", "severity_confusion_unseen_type.png")]
    with st.expander("Evaluation charts"):
        cols = st.columns(2)
        for i, (title, f) in enumerate(figs):
            p = path("figures_dir") / f
            if p.exists():
                with cols[i % 2]:
                    st.markdown(f"**{title}**")
                    st.image(str(p), use_container_width=True)
    if ho:
        with st.expander("Independent hold-out: 4 CWRU recordings never used for anything"):
            st.dataframe(pd.DataFrame(ho["results"]), hide_index=True, use_container_width=True)

import streamlit as st

from components.common import REPORT_KEY, disclaimer, input_panel, show_gauge

st.title("Bearing Health Monitor")
st.caption("Vibration-based fault diagnosis · Edge AI · Reinforcement-learning maintenance advice · AI assistant. "
           "Target application: PA (Primary Air) fan bearings — trained on the CWRU laboratory dataset.")

input_panel()

rep = st.session_state.get(REPORT_KEY)
if rep is None:
    with st.expander("What happens when I click Analyze?"):
        st.markdown(
            "1. The signal is resampled to 12 kHz and cut into short windows.\n"
            "2. 37 measurements are computed per window (how spiky, which frequencies, which knock rates).\n"
            "3. A **Random Forest** gives its opinion on the fault, and a **physics check** looks for a clear knock "
            "signature at a fault's defect frequency.\n"
            "4. A second model estimates the damage size; written rules turn it into **Healthy / Warning / Critical**.\n"
            "5. A **reinforcement-learning agent** (trained in a simulator) recommends an action.\n"
            "6. Everything lands in one report that the other pages and the AI assistant explain.")
else:
    d, h, r, w = rep.diagnosis, rep.health, rep.recommendation, rep.whatif_continue
    st.markdown("#### 2 · Results")
    st.caption(f"`{rep.input['source']}` · {rep.input['duration_s']} s · {rep.input['rpm']:g} RPM · "
               f"{rep.input['load_hp']:g} HP · {d['n_windows']} windows")
    for warn in rep.warnings:
        st.warning(warn, icon="⚠️")
    c1, c2, c3 = st.columns([1.15, 1, 1])
    with c1, st.container(border=True):
        st.markdown("**Machine condition**")
        show_gauge(h["condition_index"], h["status"], h["stage"], h["low_confidence"])
    with c2, st.container(border=True):
        st.markdown("**Diagnosis**")
        st.metric("Fault type", d["fault"])
        st.caption(f"Decided by: **{d['decision_source']}**")
        if d["decision_source"].startswith("knock"):
            st.caption(f"ML model: {d['ml_fault']} ({d['ml_confidence']:.0%}) — "
                       f"{'agrees' if d['methods_agree'] else 'disagrees'}")
        else:
            st.caption(f"ML vote share: {d['confidence']:.0%} · no clear knock signature")
        st.metric("Estimated fault size", f'{d["size_in"]:.3f}"' if d["size_in"] else "—")
        if d["size_in"]:
            st.caption("Size estimates are indicative only (see Diagnosis page).")
    with c3, st.container(border=True):
        st.markdown("**Recommended action**")
        if r:
            st.metric("RL agent (simulation-trained)", r["action"])
            if r["shielded"]:
                st.caption(f"🛡️ **Safety shield applied** — the agent chose *{r['agent_action']}*, which is not "
                           "allowed on a Critical rating, so the rule-based action is recommended.")
            else:
                st.caption(f"{r['action_probabilities'][r['action']]:.0%} of the agent's choice · rule-based policy: "
                           f"**{r['rule_based_action']}** ({'agrees' if r['agrees_with_rule'] else 'differs'})")
        if w:
            st.metric("Breakdown risk if we keep running", f"{w['p_failure_within_30_days']:.0%}",
                      help="Chance of a breakdown within 30 days in simulated futures (illustrative assumptions).")
            st.caption("within 30 days · simulated")
    st.markdown("#### 3 · Explore the details")
    cols = st.columns(5)
    for col, (pg, label, icon) in zip(cols, [("views/signal.py", "Signal", "📈"),
                                             ("views/diagnosis.py", "Diagnosis", "🩺"),
                                             ("views/maintenance.py", "Maintenance", "🛠️"),
                                             ("views/assistant.py", "Ask the AI", "💬"),
                                             ("views/edge.py", "Edge AI", "📟")]):
        with col:
            st.page_link(pg, label=label, icon=icon, use_container_width=True)

disclaimer()

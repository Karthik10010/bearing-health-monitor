import pandas as pd
import plotly.express as px
import streamlit as st

from components.common import get_report, badge, metrics, simulation_banner
from pdm.config import path

st.title("Predictive Maintenance")
rep = get_report()
simulation_banner()
if rep is not None:
    d, h, r, w = rep.diagnosis, rep.health, rep.recommendation, rep.whatif_continue
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.subheader("Machine state (what the agent sees)")
        st.html(badge(h["status"], h["low_confidence"]))
        size_txt = f'{d["size_in"]:.3f}"' if d["size_in"] else "—"
        st.write(f"Diagnosis **{d['fault']}** ({d['confidence']:.0%}), size {size_txt}, "
                 f"load {rep.input['load_hp']:g} HP")
        if r:
            st.caption("Assumptions: " + "; ".join(r["assumptions"]))
    with c2:
        if r:
            st.subheader(f"Recommended: {r['action']}")
            if r["shielded"]:
                st.warning(f"🛡️ **Safety shield applied.** The agent chose *{r['agent_action']}*, but on a Critical "
                           f"rating the agent may not recommend continuing or only monitoring, so the rule-based "
                           f"action (**{r['rule_based_action']}**) is recommended. Why the agent distrusts Critical "
                           "labels: in simulation most of them were false alarms, because size estimates are "
                           "unreliable on unseen bearings.", icon="🛡️")
            else:
                st.write(f"Agent's choice: **{r['agent_action']}** · rule-based policy: **{r['rule_based_action']}** — "
                         f"{'they agree' if r['agrees_with_rule'] else 'they differ'}.")
            st.caption("The agent's own action probabilities:")
            ap = r["action_probabilities"]
            fig = px.bar(x=list(ap.values()), y=list(ap), orientation="h", range_x=[0, 1], height=230,
                         labels={"x": "probability", "y": ""}, text=[f"{v:.0%}" for v in ap.values()])
            fig.update_layout(margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("The RL agent has not been trained yet.")
    if w:
        st.subheader("What if we keep running with no maintenance? (simulated)")
        k = st.columns(4)
        for col, days in zip(k, (7, 30, 90, w["horizon_days"])):
            col.metric(f"Breakdown within {days} days", f"{w[f'p_failure_within_{days}_days']:.0%}")
        st.caption(f"{w['n_rollouts']} simulated futures starting from this diagnosis, at {w['load_hp']} HP.")

st.divider()
st.header("How the agent was trained and how it compares (simulation)")
rl = metrics("rl_evaluation.json")
if rl:
    s = pd.DataFrame(rl["summary"]).T
    main = s[~s.index.str.startswith("PPO seed")]
    st.dataframe(main[["mean_reward", "failures_per_episode", "maintenances_per_episode", "inspections_per_episode",
                       "unnecessary_maintenance_per_episode", "downtime_days", "availability_pct"]].round(2),
                 use_container_width=True)
    pr, seeds = rl["ppo_minus_rule_paired"], rl["ppo_all_seeds_mean_reward"]
    ps = rl.get("shielded_minus_rule_paired")
    st.markdown(f"On the same {rl['test_episodes']} simulated test episodes the PPO agent (best of "
                f"{len(seeds['values'])} training seeds, chosen on separate validation episodes) earned "
                f"**{pr['mean_diff']:+.0f}** reward per episode vs. the rule-based policy "
                f"(95% interval {pr['ci95'][0]:+.0f} to {pr['ci95'][1]:+.0f})"
                + (f"; **with the safety shield** (what the dashboard recommends) **{ps['mean_diff']:+.0f}** "
                   f"({ps['ci95'][0]:+.0f} to {ps['ci95'][1]:+.0f})." if ps else "."))
    st.caption(f"The {len(seeds['values'])} trained agents (different random seeds) scored "
               + ", ".join(f"{v:.0f}" for v in seeds["values"]) + f" (mean {seeds['mean']:.0f} ± {seeds['std']:.0f}).")
for title, f in [("Training curve", "rl_training_curve.png"), ("Policy comparison", "rl_policy_comparison.png"),
                 ("Episode where PPO and the rules differ most", "rl_episode_timeline.png")]:
    p = path("figures_dir") / f
    if p.exists():
        st.markdown(f"**{title}**")
        st.image(str(p), use_container_width=True)
ex = path("metrics_dir") / "rl_example_decisions.csv"
if ex.exists():
    st.markdown("**Example decisions: agent's choice, final action with the safety shield, and the rules**")
    st.dataframe(pd.read_csv(ex), hide_index=True, use_container_width=True)

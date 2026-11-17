"""Template answers filled ONLY with values from the report context (used when there is no API key,
no internet, or the Gemini call fails). Every number below comes from the context dict."""
from __future__ import annotations

QUESTION_TYPES = {
    "condition": "What is the bearing condition?",
    "why_fault": "Why do you think it's this fault?",
    "action": "What action do you recommend?",
    "why_rl": "Why did the RL agent choose this action?",
    "whatif": "What happens if we keep running?",
    "simple": "Explain this to a non-engineer.",
    "compare": "Compare with a healthy bearing.",
}

_KEYWORDS = [
    ("whatif", ("what if", "what happens", "keep running", "continue running", "if we continue", "risk")),
    ("why_rl", ("rl", "agent", "ppo", "why did", "why does it recommend", "why recommend")),
    ("why_fault", ("why", "how do you know", "evidence", "reason")),
    ("compare", ("compare", "healthy bearing", "normal bearing", "difference")),
    ("simple", ("simple", "non-engineer", "layman", "easy", "eli5", "plain")),
    ("action", ("action", "recommend", "should we", "what to do", "next step", "do now")),
    ("condition", ("condition", "status", "health", "state", "how is", "ok?", "okay")),
]

DEFECT_NAMES = {"bpfo": "outer race (BPFO)", "bpfi": "inner race (BPFI)", "bsf": "ball spin (BSF)"}


def classify(question: str) -> str:
    q = question.lower()
    for qtype, words in _KEYWORDS:
        if any(w in q for w in words):
            return qtype
    return "condition"


def _pct(x) -> str:
    return f"{100 * x:.0f}%"


def _size(d) -> str:
    return f'{d["size_in"]:.3f}"' if d.get("size_in") is not None else "not applicable (no fault predicted)"


def _strongest_peak(ctx) -> tuple[str, dict]:
    peaks = {k: v for k, v in ctx["envelope_peaks"].items() if k in DEFECT_NAMES}
    k = max(peaks, key=lambda n: peaks[n]["peak_ratio_h1"])
    return k, peaks[k]


def _source_line(d) -> str:
    ke = d["knock_evidence"]
    if ke["call"]:
        agree = "the ML model agrees" if d["methods_agree"] else f"the ML model said {d['ml_fault']} instead"
        return (f"clear {ke['call'].lower()} knock signature: {ke['peak_ratios'][ke['call']]}x the background, "
                f"{ke['dominance']}x the next fault family; {agree}")
    return "no clear knock signature, so the ML model decides"


def _caveat_line(ctx) -> str:
    ev = ctx["model_evaluation"]["fault_type_unseen_bearings"]
    hy = ctx["model_evaluation"].get("hybrid_unseen_bearings")
    combo = (f" Combined with the physics knock check, {_pct(hy['hybrid_accuracy'])} of unseen-bearing recordings "
             f"were typed correctly." if hy else "")
    return (f"Note: on bearings it has never seen, the ML model alone typed {_pct(ev['recording_accuracy'])} of "
            f"recordings correctly (macro-F1 {ev['window_macro_f1']}).{combo} Healthy vs damaged is more reliable: "
            f"{_pct(ev['normal_windows_correct'])} of healthy windows were recognised.")


def answer(qtype: str, ctx: dict) -> str:
    d, h, rec = ctx["diagnosis"], ctx["health"], ctx.get("recommendation")
    probs = ", ".join(f"{k} {_pct(v)}" for k, v in sorted(d["probabilities"].items(), key=lambda kv: -kv[1]))
    flag = " The confidence is low, so the system recommends re-measuring." if h["low_confidence"] else ""
    warn = (" Input warnings: " + "; ".join(ctx["warnings"]) + ".") if ctx.get("warnings") else ""

    if qtype == "condition":
        return (f"Diagnosis: **{d['fault']}**, decided by the {d['decision_source']} ({_source_line(d)}). "
                f"ML model vote shares: {probs}. "
                f"Estimated fault size: {_size(d)}. Health status: **{h['status']}** ({h['stage']}) - "
                f"{h['label'].lower()}: {'; '.join(h['reasons'])}.{flag}{warn}\n\n{_caveat_line(ctx)}")

    if qtype == "why_fault":
        k, p = _strongest_peak(ctx)
        feats = "; ".join(f"{f['feature']} = {f['value']} (healthy median {f['healthy_median']}"
                          + (f", {f['ratio_to_healthy']}x" if f["ratio_to_healthy"] is not None else "") + ")"
                          for f in ctx["top_features"][:4])
        if p["peak_ratio_h1"] >= 3:
            env_txt = (f"Envelope spectrum: the strongest defect-frequency peak is the {DEFECT_NAMES[k]} at "
                       f"{p['frequency_hz']} Hz, {p['peak_ratio_h1']}x the background level (healthy bearings: "
                       f"about {p['healthy_median_h1']}x).")
        else:
            env_txt = (f"Envelope spectrum: there is NO clear defect-frequency peak (the highest is the "
                       f"{DEFECT_NAMES[k]} at {p['peak_ratio_h1']}x the background; healthy bearings show about "
                       f"{p['healthy_median_h1']}x), so the decision rests mainly on the overall vibration "
                       f"measurements below rather than on a textbook defect signature.")
        return (f"Two independent opinions were combined. The Random Forest's trees voted {probs}; "
                f"{_pct(d['window_agreement'])} of the {d['n_windows']} signal windows agreed with "
                f"**{d['fault']}**. Final decision by the {d['decision_source']} ({_source_line(d)}).\n\n"
                f"{env_txt} Most important measurements: {feats}.\n\n{_caveat_line(ctx)}")

    if qtype == "action":
        if not rec:
            return "No recommendation is available: the RL agent has not been trained yet."
        if rec["shielded"]:
            head = (f"Recommended: **{rec['action']}**. The RL agent itself chose {rec['agent_action']}, but a "
                    f"safety shield does not allow continuing or only monitoring on a Critical rating, so the "
                    f"rule-based action is used.")
        else:
            head = (f"The RL agent recommends: **{rec['action']}** (it gives this action "
                    f"{_pct(rec['action_probabilities'][rec['action']])} probability). The rule-based policy says: "
                    f"**{rec['rule_based_action']}** ({'same' if rec['agrees_with_rule'] else 'different'}).")
        return (f"{head} Health status: {h['status']} ({h['label'].lower()}).\n\n{rec['label']}")

    if qtype == "why_rl":
        if not rec:
            return "No recommendation is available: the RL agent has not been trained yet."
        ap = ", ".join(f"{k} {_pct(v)}" for k, v in sorted(rec["action_probabilities"].items(), key=lambda kv: -kv[1]))
        rl = ctx["model_evaluation"].get("rl_simulated_test", {})
        perf = ""
        key = "PPO agent + safety shield" if "PPO agent + safety shield" in rl else "PPO agent"
        if key in rl and "Rule-based" in rl:
            perf = (f" In simulated tests the {key.lower()} averaged {rl[key]['mean_reward']} reward per 180-day "
                    f"episode vs {rl['Rule-based']['mean_reward']} for the rule-based policy, with "
                    f"{rl[key]['failures_per_episode']} vs {rl['Rule-based']['failures_per_episode']} "
                    f"breakdowns per episode.")
        shield_txt = (f" Its own top choice was {rec['agent_action']}, which the safety shield replaced because the "
                      f"rating is Critical (in simulation most Critical labels were false alarms, so the agent had "
                      f"learned to distrust them)." if rec["shielded"] else "")
        return (f"The agent saw: diagnosis {d['fault']} ({_pct(d['confidence'])}), estimated size {_size(d)}, "
                f"health {h['status']}, load {ctx['input']['load_hp']} HP. It learned by trial and error in a "
                f"simulator where running earns money, inspections/repairs cost a little and breakdowns cost a "
                f"lot. Its action probabilities: {ap}.{shield_txt}{perf} Assumptions: "
                f"{'; '.join(rec['assumptions'])}.\n\n"
                f"{rec['label']}")

    if qtype == "whatif":
        w = ctx.get("whatif_continue")
        if not w:
            return "The what-if simulation was not run for this analysis."
        med = w.get("median_days_to_failure_if_fails")
        return (f"SIMULATED, keeping the machine running at {w['load_hp']} HP with no maintenance "
                f"({w['n_rollouts']} simulated futures): chance of a breakdown within 7 days "
                f"{_pct(w['p_failure_within_7_days'])}, within 30 days {_pct(w['p_failure_within_30_days'])}, "
                f"within 90 days {_pct(w['p_failure_within_90_days'])}"
                + (f"; when it does break down, the median is day {med:.0f}." if med else ".")
                + " These numbers come from the simulator's assumed wear rates and costs, not from plant data.")

    if qtype == "simple":
        status_words = {"Healthy": "looks healthy", "Warning": "shows signs of damage that should be watched",
                        "Critical": "shows serious damage"}
        what = ("no damage" if d["fault"] == "Normal"
                else f"damage on the {d['fault'].lower()} part of the bearing (estimated size {_size(d)})")
        act = f" The suggested next step is: {rec['action'].lower()}." if rec else ""
        return (f"Think of the bearing as a ring of steel balls that lets the shaft spin. A crack makes a tiny "
                f"'knock' every time a ball rolls over it, and the sensor hears those knocks. This recording "
                f"suggests {what}, and the bearing {status_words[h['status']]} (the system is "
                f"{_pct(d['confidence'])} sure).{act} This is an automated estimate from lab-trained models, so "
                f"a person should confirm it.")

    if qtype == "compare":
        rows = "\n".join(f"- {f['feature']}: {f['value']} vs healthy {f['healthy_median']} "
                         f"(typical healthy range {f['healthy_p25_p75'][0]} to {f['healthy_p25_p75'][1]})"
                         for f in ctx["top_features"][:6])
        return f"Compared with healthy CWRU bearings (median of the training data):\n{rows}"

    return answer("condition", ctx)

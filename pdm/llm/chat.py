"""Chat orchestration: pick Gemini or the offline fallback, add on-demand simulations, run the guard."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from pdm.llm import fallback, gemini_client, guard
from pdm.llm.context import build_context, context_json


@dataclass
class ChatAnswer:
    text: str
    mode: str                     # "gemini" | "fallback"
    question_type: str
    unsupported_numbers: list[str] = field(default_factory=list)
    context: dict = field(default_factory=dict)
    note: str | None = None       # e.g. why we fell back

    def to_dict(self) -> dict:
        return asdict(self)


def _extra_simulations(question: str, report) -> dict | None:
    """'What if we reduce the load?' -> actually run the simulator with that action."""
    q = question.lower()
    if fallback.classify(question) == "whatif" and ("reduce" in q or "lower load" in q):
        from pdm.pipeline import _obs_model
        from pdm.rl.whatif import simulate_continue
        d = report.diagnosis
        res = simulate_continue(d["probabilities"], d["size_in"], int(min(3, max(0, round(report.input["load_hp"])))),
                                n_rollouts=300, action=3, obs_model=_obs_model())
        res["policy"] = "reduce operating load every day"
        return {"whatif_reduce_load": res}
    return None


def answer(question: str, report, history: list[dict] | None = None, use_gemini: bool | None = None) -> ChatAnswer:
    qtype = fallback.classify(question)
    ctx = build_context(report, _extra_simulations(question, report))
    want_gemini = gemini_client.available() if use_gemini is None else use_gemini
    note = None
    if want_gemini:
        try:
            text = gemini_client.ask(question, context_json(ctx), history)
            return ChatAnswer(text, "gemini", qtype, guard.unsupported_numbers(text, ctx, question), ctx)
        except Exception as exc:  # network / quota / bad key -> keep the demo working
            note = f"Gemini unavailable ({type(exc).__name__}: {str(exc)[:160]}); showing the offline answer."
    elif use_gemini is None:
        note = "No GEMINI_API_KEY set - offline template answers (real report values) are shown."
    if qtype == "whatif" and "additional_simulations" in ctx:
        w = ctx["additional_simulations"]["whatif_reduce_load"]
        text = (fallback.answer("whatif", ctx) + f"\n\nSIMULATED with REDUCED load every day instead: breakdown "
                f"within 30 days {100 * w['p_failure_within_30_days']:.0f}%, within 90 days "
                f"{100 * w['p_failure_within_90_days']:.0f}%.")
    else:
        text = fallback.answer(qtype, ctx)
    return ChatAnswer(text, "fallback", qtype, guard.unsupported_numbers(text, ctx, question), ctx, note)

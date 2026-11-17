"""Google Gemini calls (official google-genai SDK). The key is read from .env (GEMINI_API_KEY)."""
from __future__ import annotations

import os

from pdm.config import PROJECT_ROOT, load_config

SYSTEM_PROMPT = """You are the maintenance assistant of a bearing condition-monitoring system.
You explain ONE analysis report, given below as JSON, to plant engineers.

STRICT RULES
1. Use ONLY facts and numbers that appear in the REPORT JSON. Never invent sensor readings, predictions,
   probabilities, costs, dates or maintenance actions. Do not do new calculations beyond simple rounding.
2. If the answer is not in the report, say "That is not available in this report."
3. The maintenance recommendation, action probabilities, rewards and all what-if / failure-risk numbers come
   from a SIMULATION with assumed costs: always call them "simulated".
4. The health status is a "rule-based estimate". The models were trained on CWRU laboratory data, not on a
   real PA fan; mention this when it matters. When you talk about how sure the fault type is, mention that
   accuracy on bearings the model has never seen is much lower (model_evaluation.fault_type_unseen_bearings).
5. Do not recommend an action other than the report's RL action or rule-based action.
6. Be concise (under 200 words unless asked for more), use plain language, and define any technical term.
"""


def api_key() -> str | None:
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except ImportError:
        pass
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    return key or None


def available() -> bool:
    return api_key() is not None


def ask(question: str, context_json: str, history: list[dict] | None = None, timeout_s: float = 60.0) -> str:
    """history: [{"role": "user"|"assistant", "content": str}, ...] (previous turns only)."""
    from google import genai
    from google.genai import types

    key = api_key()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set (see .env.example)")
    cfg = load_config()["llm"]
    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))
    contents = []
    for turn in history or []:
        role = "model" if turn["role"] == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=turn["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=question)]))
    resp = client.models.generate_content(
        model=cfg["gemini_model"], contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT + "\nREPORT JSON:\n" + context_json,
            temperature=cfg["temperature"], max_output_tokens=cfg["max_output_tokens"]),
    )
    text = (resp.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty answer")
    return text

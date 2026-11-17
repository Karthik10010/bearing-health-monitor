"""AnalysisReport -> compact JSON context. This is the ONLY data the language model receives."""
from __future__ import annotations

import json
from typing import Any


def _round(o: Any, nd: int = 4) -> Any:
    if isinstance(o, float):
        return round(o, nd)
    if isinstance(o, dict):
        return {k: _round(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_round(v, nd) for v in o]
    return o


def build_context(report, extra: dict | None = None) -> dict:
    """Everything from the report except raw arrays (plot data), plus optional extra simulator results."""
    d = report.to_dict(include_arrays=False)
    if d.get("recommendation"):
        d["recommendation"].pop("observation", None)  # raw vector - not useful for explanations
    if extra:
        d["additional_simulations"] = extra
    return _round(d)


def context_json(ctx: dict) -> str:
    return json.dumps(ctx, indent=1, ensure_ascii=False)

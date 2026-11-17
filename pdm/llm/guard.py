"""Check that every number in an AI answer actually appears in the data it was given."""
from __future__ import annotations

import re
from typing import Any

_NUM = re.compile(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?%?")


def _context_numbers(o: Any, out: set[float]) -> None:
    if isinstance(o, bool):
        return
    if isinstance(o, (int, float)):
        out.add(float(o))
        out.add(float(o) * 100)  # 0.94 may be written as 94%
    elif isinstance(o, str):
        for m in _NUM.findall(o.replace("_", " ")):  # JSON keys like p_failure_within_90_days
            out.add(float(m.rstrip("%").replace(",", "")))
    elif isinstance(o, dict):
        for k, v in o.items():
            _context_numbers(k, out)
            _context_numbers(v, out)
    elif isinstance(o, (list, tuple)):
        for v in o:
            _context_numbers(v, out)


def unsupported_numbers(answer: str, context: dict, question: str = "", ignore_small_ints: int = 10) -> list[str]:
    """Numbers in `answer` that cannot be matched to the context (allowing for rounding).

    Small integers (<= 10) are ignored: they are usually counts or ordinals ("3 harmonics", "1x").
    """
    known: set[float] = set()
    _context_numbers(context, known)
    _context_numbers(question, known)
    bad = []
    for tok in _NUM.findall(answer):
        s = tok.rstrip("%").replace(",", "")
        n = float(s)
        if n.is_integer() and abs(n) <= ignore_small_ints and "." not in s:
            continue
        decimals = len(s.split(".")[1]) if "." in s else 0
        tol = 0.5 * 10 ** (-decimals) + 1e-9
        if not any(abs(n - v) <= tol for v in known):
            bad.append(tok)
    return bad

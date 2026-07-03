"""Program-of-thought support: extract and safely execute LLM-emitted arithmetic.

FinanceBench-style questions frequently require *computation* (margins, YoY deltas,
ratios), where single-shot text generation reliably fumbles arithmetic. The generator
may emit a fenced ```python block that computes a variable named ``answer``; we execute
it in a restricted sandbox and feed the result back for the final grounded answer.

The sandbox is deliberately tiny: pure arithmetic over literals. No imports, no
dunder access, no I/O, bounded size and (nominal) loop constructs rejected.
"""

from __future__ import annotations

import re

_CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)

# Reject anything that could escape pure arithmetic.
_FORBIDDEN = re.compile(
    r"(__|\bimport\b|\bopen\b|\bexec\b|\beval\b|\bglobals\b|\blocals\b|\bgetattr\b"
    r"|\bsetattr\b|\bdelattr\b|\bwhile\b|\bclass\b|\blambda\b|\byield\b|\binput\b)"
)

_SAFE_BUILTINS = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "len": len, "float": float, "int": int, "range": range, "sorted": sorted,
}


def extract_program(text: str) -> str | None:
    """Return the first fenced python block, if any."""
    m = _CODE_FENCE.search(text or "")
    return m.group(1).strip() if m else None


def run_program(code: str) -> tuple[str | None, str | None]:
    """Execute ``code`` in the arithmetic sandbox. Returns (result, error)."""
    if not code or len(code) > 2000:
        return None, "program rejected: empty or too long"
    if _FORBIDDEN.search(code):
        return None, "program rejected: forbidden construct"
    scope: dict = {}
    try:
        exec(compile(code, "<pot>", "exec"), {"__builtins__": _SAFE_BUILTINS}, scope)  # noqa: S102
    except Exception as exc:
        return None, f"program error: {exc}"
    if "answer" not in scope:
        return None, "program did not set `answer`"
    value = scope["answer"]
    if isinstance(value, float):
        value = round(value, 6)
    return str(value), None

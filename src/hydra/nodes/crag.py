"""Phase 4 — Corrective RAG (CRAG) quality gate.

A retrieval evaluator grades the candidates *before* they reach the generator, and a
router acts on that grade: high confidence generates directly, medium/low triggers a
bounded secondary-retrieval loop, and exhausted low confidence asks the user to clarify
rather than answering on weak evidence.
"""

from __future__ import annotations

from hydra.config import Settings
from hydra.llm import LLMClient, parse_json

EVAL_SYSTEM = """You are a retrieval quality evaluator for a RAG system. Given a user
question and the retrieved context passages, judge whether the context is sufficient and
relevant to answer the question. Grade as:
- "high": context clearly and directly supports a complete answer.
- "medium": context is partially relevant; an answer is possible but may be incomplete.
- "low": context is off-topic or insufficient.

Respond with ONLY JSON:
{"confidence": "high"|"medium"|"low", "score": <0.0-1.0>, "reasoning": "<one sentence>"}

TASK: retrieval_eval"""


def _format_context(candidates: list[dict], limit: int = 5) -> str:
    lines = []
    for c in candidates[:limit]:
        meta = c.get("metadata", {})
        loc = f"{meta.get('source', '?')} p.{meta.get('page', '?')} sec.{meta.get('section', '?')}"
        lines.append(f"- ({loc}) {c.get('text', '')}")
    return "\n".join(lines)


def make_retrieval_evaluator(llm: LLMClient, settings: Settings):
    """Build the ``retrieval_evaluator`` node (LLM grader)."""

    def retrieval_evaluator(state: dict) -> dict:
        candidates = state.get("candidates", []) or []
        if not candidates:
            # Nothing to grade — short-circuit to low without spending an LLM call.
            return {
                "retrieval_confidence": "low",
                "retrieval_score": 0.0,
                "trace": [{"node": "retrieval_evaluator", "detail": "low (no candidates)"}],
            }

        user = f"Question: {state['query']}\n\nContext:\n{_format_context(candidates, settings.generation_context_k)}"
        data = parse_json(llm.complete(system=EVAL_SYSTEM, user=user)) or {}
        confidence = data.get("confidence")
        if confidence not in ("high", "medium", "low"):
            confidence = "medium"  # unparseable -> neither trust nor discard
        try:
            score = float(data.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0

        return {
            "retrieval_confidence": confidence,
            "retrieval_score": score,
            "trace": [{"node": "retrieval_evaluator", "detail": f"{confidence} (score={score:.2f})"}],
        }

    return retrieval_evaluator


def make_secondary_retrieve(retriever=None, *, settings: Settings):
    """Build the ``secondary_retrieve`` node — broadened retrieval on weak confidence."""

    def secondary_retrieve(state: dict) -> dict:
        retries = state.get("crag_retries", 0) + 1
        queries = state.get("search_queries") or [state["query"]]
        if retriever is None:
            return {
                "crag_retries": retries,
                "trace": [{"node": "secondary_retrieve",
                           "detail": f"attempt {retries}: no retriever attached"}],
            }
        broadened_k = settings.retrieval_top_k * 2
        results = retriever.retrieve(
            queries,
            original_query=state["query"],
            hyde_doc=state.get("hyde_doc") or None,
            top_k=broadened_k,
        )
        return {
            "crag_retries": retries,
            "candidates": [r.as_dict() for r in results],
            "trace": [{"node": "secondary_retrieve",
                       "detail": f"attempt {retries}: broadened to top {broadened_k} -> {len(results)}"}],
        }

    return secondary_retrieve


def ask_user(state: dict) -> dict:
    """Terminal node when retrieval stays low-confidence: ask for clarification."""
    return {
        "answer": "I couldn't find sufficiently relevant information to answer confidently. "
                  "Could you rephrase or add detail?",
        "citations": [],
        "trace": [{"node": "ask_user", "detail": "low confidence after retries -> clarification"}],
    }


def make_crag_router(settings: Settings):
    """Conditional-edge fn: generate | secondary_retrieve | ask_user."""

    def crag_router(state: dict) -> str:
        confidence = state.get("retrieval_confidence", "low")
        retries = state.get("crag_retries", 0)
        if confidence == "high":
            return "generate"
        if retries >= settings.crag_max_retries:
            # Out of retries: answer on partial evidence, or bail out if truly weak.
            return "generate" if confidence == "medium" else "ask_user"
        return "secondary_retrieve"

    return crag_router

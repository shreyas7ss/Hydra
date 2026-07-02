"""Phase 4 — grounded generation + Self-RAG reflection.

``generate`` answers strictly from retrieved context and emits citations (page/section/
node IDs) for auditability. ``self_rag_reflect`` is an LLM self-critique that emits
reflection tokens for faithfulness + relevance; if the answer isn't faithful, the router
loops back for a bounded number of regenerations.
"""

from __future__ import annotations

from hydra.config import Settings
from hydra.llm import LLMClient, parse_json

GENERATE_SYSTEM = """You are a careful RAG assistant. Answer the question using ONLY the
provided context. If the context is insufficient, say so plainly. Be concise and do not
invent facts beyond the context.

TASK: generate"""

REFLECT_SYSTEM = """You are a Self-RAG critic. Given a question, a candidate answer, and the
context it was based on, emit reflection tokens judging the answer:
- faithful: is every claim supported by the context (no hallucination)?
- relevant: does the answer actually address the question?

Respond with ONLY JSON:
{"faithful": true|false, "relevant": true|false, "critique": "<one sentence>"}

TASK: reflect"""


def _context_block(candidates: list[dict], limit: int) -> str:
    lines = []
    for c in candidates[:limit]:
        meta = c.get("metadata", {})
        loc = f"{meta.get('source', '?')} p.{meta.get('page', '?')} sec.{meta.get('section', '?')}"
        lines.append(f"[{c.get('id', '?')} | {loc}] {c.get('text', '')}")
    return "\n".join(lines)


def _citations(candidates: list[dict], limit: int) -> list[dict]:
    citations = []
    for c in candidates[:limit]:
        meta = c.get("metadata", {})
        citations.append({
            "id": c.get("id"),
            "source": meta.get("source"),
            "page": meta.get("page"),
            "section": meta.get("section"),
        })
    return citations


def make_generate(llm: LLMClient, settings: Settings):
    """Build the ``generate`` node."""

    def generate(state: dict) -> dict:
        candidates = state.get("candidates", []) or []
        count = state.get("generation_count", 0) + 1

        if not candidates:
            return {
                "answer": "I don't have enough retrieved context to answer that.",
                "citations": [],
                "generation_count": count,
                "trace": [{"node": "generate", "detail": f"attempt {count}: no context"}],
            }

        k = settings.generation_context_k
        user = f"Question: {state['query']}\n\nContext:\n{_context_block(candidates, k)}"
        answer = llm.complete(system=GENERATE_SYSTEM, user=user).strip()
        return {
            "answer": answer,
            "citations": _citations(candidates, k),
            "generation_count": count,
            "trace": [{"node": "generate", "detail": f"attempt {count}: {len(answer)} chars, {min(k, len(candidates))} cites"}],
        }

    return generate


def make_self_rag_reflect(llm: LLMClient, settings: Settings):
    """Build the ``self_rag_reflect`` node (LLM self-critique)."""

    def self_rag_reflect(state: dict) -> dict:
        candidates = state.get("candidates", []) or []
        answer = state.get("answer", "")
        user = (
            f"Question: {state['query']}\n\n"
            f"Answer: {answer}\n\n"
            f"Context:\n{_context_block(candidates, settings.generation_context_k)}"
        )
        data = parse_json(llm.complete(system=REFLECT_SYSTEM, user=user)) or {}
        faithful = bool(data.get("faithful", False))
        relevant = bool(data.get("relevant", False))
        reflection = {
            "faithful": faithful,
            "relevant": relevant,
            "ok": faithful and relevant,
            "critique": str(data.get("critique", "")).strip(),
        }
        return {
            "reflection": reflection,
            "trace": [{"node": "self_rag_reflect",
                       "detail": f"faithful={faithful} relevant={relevant}"}],
        }

    return self_rag_reflect


def make_reflect_router(settings: Settings):
    """Conditional-edge fn: end | regenerate (bounded by generation_count)."""

    def reflect_router(state: dict) -> str:
        reflection = state.get("reflection", {})
        if reflection.get("ok"):
            return "end"
        if state.get("generation_count", 0) > settings.reflect_max_retries:
            return "end"  # give up regenerating; return best effort with its critique
        return "regenerate"

    return reflect_router

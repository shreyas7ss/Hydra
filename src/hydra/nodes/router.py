"""Phase 1 — the Intent Classification Gate.

Routes a query between the fast direct-lookup path and the agentic reasoning path.
The conditional edge function (``intent_router``) reads the classification the node
wrote to state; keeping classification (a node, side-effecting state) separate from
routing (a pure edge function) is the idiomatic LangGraph split.
"""

from __future__ import annotations

from hydra.config import Settings
from hydra.llm import LLMClient, parse_json

INTENT_SYSTEM = """You are the intent-classification gate of a hybrid RAG system.
Classify the user's query into exactly one of:

- "direct": a simple factoid, keyword, or exact-identifier lookup (a specific clause
  number, SKU, date, or single value) best served by a fast BM25/SQL lookup.
- "complex": an ambiguous, comparative, or multi-step / multi-hop question that needs
  reasoning over multiple passages or documents.

Respond with ONLY a JSON object:
{"intent": "direct" | "complex", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}

TASK: intent_classification"""


def make_route_intent(llm: LLMClient, settings: Settings):
    """Build the ``route_intent`` node bound to an LLM + settings."""

    def route_intent(state: dict) -> dict:
        query = state["query"]
        raw = llm.complete(system=INTENT_SYSTEM, user=query)
        data = parse_json(raw) or {}

        intent = data.get("intent")
        try:
            confidence = float(data.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        reasoning = str(data.get("reasoning", "")).strip()

        # Unparseable / invalid -> default to the thorough path (never under-serve).
        if intent not in ("direct", "complex"):
            intent = "complex"
            reasoning = reasoning or "defaulted to complex (unparseable classification)"

        # Low-confidence "direct" calls are escalated to the complex path.
        if intent == "direct" and confidence < settings.intent_confidence_floor:
            reasoning = f"escalated: low confidence {confidence:.2f}. {reasoning}".strip()
            intent = "complex"

        return {
            "intent": intent,
            "intent_confidence": confidence,
            "intent_reasoning": reasoning,
            "trace": [{"node": "route_intent", "detail": f"{intent} (conf={confidence:.2f})"}],
        }

    return route_intent


def intent_router(state: dict) -> str:
    """Conditional-edge function: returns the intent key used to pick the next node."""
    return state.get("intent", "complex")

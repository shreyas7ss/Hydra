"""The shared graph state — the single contract every LangGraph node reads/writes.

Keeping this deliberately broad: Phase 1 only populates the routing + transform
fields, but the downstream (retrieval/generation/audit) fields are declared now so
later phases bolt on without reshaping state.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Any, Literal, TypedDict

# The Intent Classification Gate decides between two retrieval philosophies:
#   "direct"  -> fast BM25/SQL lookup (minimise the Latency Tax)
#   "complex" -> query transform + hybrid/vectorless reasoning path
Intent = Literal["direct", "complex"]


class TraceEntry(TypedDict):
    """One step in the audit trail. The accumulated trace is the per-query
    reasoning path the regulated-environment objective requires."""

    node: str
    detail: str


class RAGState(TypedDict, total=False):
    # --- Input ---
    query: str

    # --- Phase 1: routing (route_intent) ---
    intent: Intent
    intent_confidence: float
    intent_reasoning: str

    # --- Phase 1: query transformation (transform_query) ---
    expanded_queries: list[str]  # Multi-Query Expansion variants
    sub_queries: list[str]       # Sub-Query Decomposition (multi-hop)
    hyde_doc: str                # Hypothetical Document — an embedding seed for Phase 2
    search_queries: list[str]    # final fan-out fed to retrieval (deduped)

    # --- Downstream (stubbed until Phase 2/3) ---
    retrieval_path: str
    candidates: list[dict[str, Any]]
    answer: str
    citations: list[dict[str, Any]]

    # --- Audit trail (appended across every node via the `add` reducer) ---
    trace: Annotated[list[TraceEntry], add]

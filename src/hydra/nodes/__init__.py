"""LangGraph nodes for Hydra.

Each public factory takes the injected ``llm`` / ``settings`` (and, for retrieval
nodes, a ``HybridRetriever``) and returns a single-argument ``(state) -> dict`` callable,
which is exactly the shape LangGraph expects for a node. Conditional-edge functions
(``*_router``) are pure and read what the preceding node wrote to state.
"""

from hydra.nodes.crag import (
    ask_user,
    make_crag_router,
    make_retrieval_evaluator,
    make_secondary_retrieve,
)
from hydra.nodes.downstream import make_direct_lookup, make_hybrid_retrieve
from hydra.nodes.generate import make_generate, make_reflect_router, make_self_rag_reflect
from hydra.nodes.router import intent_router, make_route_intent
from hydra.nodes.transform import make_transform_query

__all__ = [
    # Phase 1
    "make_route_intent",
    "intent_router",
    "make_transform_query",
    # Phase 2
    "make_direct_lookup",
    "make_hybrid_retrieve",
    # Phase 4 — CRAG
    "make_retrieval_evaluator",
    "make_secondary_retrieve",
    "ask_user",
    "make_crag_router",
    # Phase 4 — generation + Self-RAG
    "make_generate",
    "make_self_rag_reflect",
    "make_reflect_router",
]

"""LangGraph nodes for Hydra.

Phase 1 nodes live here. Each public factory takes the injected ``llm`` + ``settings``
and returns a single-argument ``(state) -> dict`` callable, which is exactly the
shape LangGraph expects for a node.
"""

from hydra.nodes.downstream import direct_lookup, hybrid_retrieve
from hydra.nodes.router import intent_router, make_route_intent
from hydra.nodes.transform import make_transform_query

__all__ = [
    "make_route_intent",
    "intent_router",
    "make_transform_query",
    "direct_lookup",
    "hybrid_retrieve",
]

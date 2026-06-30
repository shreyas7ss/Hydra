r"""Assemble the Phase 1 front-end as a LangGraph state machine.

    START -> route_intent --(direct)--> direct_lookup  -> END
                          \--(complex)-> transform_query -> hybrid_retrieve -> END

Phases 2-4 extend this same graph: hybrid_retrieve gains a PageIndex branch, and a
CRAG retrieval-evaluator + Self-RAG reflection cycle is inserted before END.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from hydra.config import Settings
from hydra.llm import LLMClient
from hydra.nodes import (
    direct_lookup,
    hybrid_retrieve,
    intent_router,
    make_route_intent,
    make_transform_query,
)
from hydra.state import RAGState


def build_frontend_graph(*, llm: LLMClient, settings: Settings):
    """Compile and return the runnable Phase 1 front-end graph."""
    graph = StateGraph(RAGState)

    graph.add_node("route_intent", make_route_intent(llm, settings))
    graph.add_node("transform_query", make_transform_query(llm, settings))
    graph.add_node("direct_lookup", direct_lookup)
    graph.add_node("hybrid_retrieve", hybrid_retrieve)

    graph.add_edge(START, "route_intent")
    graph.add_conditional_edges(
        "route_intent",
        intent_router,
        {"direct": "direct_lookup", "complex": "transform_query"},
    )
    graph.add_edge("transform_query", "hybrid_retrieve")
    graph.add_edge("direct_lookup", END)
    graph.add_edge("hybrid_retrieve", END)

    return graph.compile()


def run_query(query: str, *, llm: LLMClient, settings: Settings) -> dict:
    """Convenience: run one query through the front-end and return final state."""
    app = build_frontend_graph(llm=llm, settings=settings)
    return app.invoke({"query": query})

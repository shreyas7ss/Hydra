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
    intent_router,
    make_direct_lookup,
    make_hybrid_retrieve,
    make_route_intent,
    make_transform_query,
)
from hydra.state import RAGState


def build_frontend_graph(*, llm: LLMClient, settings: Settings, retriever=None):
    """Compile and return the runnable front-end graph.

    Pass a ``HybridRetriever`` to enable real Phase 2 retrieval; omit it for the
    Phase 1 front-end with stubbed retrieval.
    """
    graph = StateGraph(RAGState)

    graph.add_node("route_intent", make_route_intent(llm, settings))
    graph.add_node("transform_query", make_transform_query(llm, settings))
    graph.add_node("direct_lookup", make_direct_lookup(retriever, top_k=settings.retrieval_top_k))
    graph.add_node("hybrid_retrieve", make_hybrid_retrieve(retriever, top_k=settings.retrieval_top_k))

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


def run_query(query: str, *, llm: LLMClient, settings: Settings, retriever=None) -> dict:
    """Convenience: run one query through the graph and return final state."""
    app = build_frontend_graph(llm=llm, settings=settings, retriever=retriever)
    return app.invoke({"query": query})

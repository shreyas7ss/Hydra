r"""Assemble the Hydra pipeline as a LangGraph state machine.

    START -> route_intent
      route_intent --(direct)--> direct_lookup ------------------> generate
      route_intent --(complex)-> transform_query -> hybrid_retrieve -> retrieval_evaluator
                                                                          | crag_router:
                                    high ---------------------------------+-> generate
                                    med/low (retries left) -> secondary_retrieve -> retrieval_evaluator
                                    med (exhausted) --------------------------------> generate
                                    low (exhausted) --------------------------------> ask_user -> END
      generate -> self_rag_reflect --(reflect_router)--> END | generate  (bounded regeneration)
      ask_user -> END

Phase 3 (PageIndex tree search) and Phase 5 (compression / Proxy-Pointer) slot in at the
retrieval_evaluator input and the pre-generate edge respectively.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from hydra.config import Settings
from hydra.llm import LLMClient
from hydra.nodes import (
    ask_user,
    intent_router,
    make_crag_router,
    make_direct_lookup,
    make_generate,
    make_hybrid_retrieve,
    make_pageindex_search,
    make_reflect_router,
    make_retrieval_evaluator,
    make_retrieval_router,
    make_route_intent,
    make_secondary_retrieve,
    make_self_rag_reflect,
    make_transform_query,
)
from hydra.pageindex.search import TreeStore
from hydra.state import RAGState


def build_graph(*, llm: LLMClient, settings: Settings, retriever=None, tree_store: TreeStore | None = None):
    """Compile and return the runnable pipeline graph.

    Pass a ``HybridRetriever`` to enable real retrieval; omit it to run the front-end
    with stubbed retrieval (the CRAG loop then resolves to a clarification via ask_user).
    Pass a ``TreeStore`` to enable the PageIndex fine-grained stage after the coarse filter.
    """
    tree_store = tree_store if tree_store is not None else TreeStore()
    graph = StateGraph(RAGState)

    # Nodes
    graph.add_node("route_intent", make_route_intent(llm, settings))
    graph.add_node("transform_query", make_transform_query(llm, settings))
    graph.add_node("direct_lookup", make_direct_lookup(retriever, top_k=settings.retrieval_top_k))
    graph.add_node("hybrid_retrieve", make_hybrid_retrieve(retriever, top_k=settings.retrieval_top_k))
    graph.add_node("pageindex_tree_search", make_pageindex_search(tree_store, llm, settings))
    graph.add_node("retrieval_evaluator", make_retrieval_evaluator(llm, settings))
    graph.add_node("secondary_retrieve", make_secondary_retrieve(retriever, settings=settings))
    graph.add_node("ask_user", ask_user)
    graph.add_node("generate", make_generate(llm, settings))
    graph.add_node("self_rag_reflect", make_self_rag_reflect(llm, settings))

    # Phase 1 routing
    graph.add_edge(START, "route_intent")
    graph.add_conditional_edges(
        "route_intent",
        intent_router,
        {"direct": "direct_lookup", "complex": "transform_query"},
    )
    graph.add_edge("transform_query", "hybrid_retrieve")

    # Direct fast path bypasses the CRAG loop; complex path goes through it.
    graph.add_edge("direct_lookup", "generate")

    # Phase 3: hybrid coarse filter -> optional PageIndex fine-grained tree search.
    graph.add_conditional_edges(
        "hybrid_retrieve",
        make_retrieval_router(tree_store, settings),
        {"pageindex": "pageindex_tree_search", "evaluate": "retrieval_evaluator"},
    )
    graph.add_edge("pageindex_tree_search", "retrieval_evaluator")

    # Phase 4 CRAG gate (cyclic: secondary_retrieve -> evaluator)
    graph.add_conditional_edges(
        "retrieval_evaluator",
        make_crag_router(settings),
        {
            "generate": "generate",
            "secondary_retrieve": "secondary_retrieve",
            "ask_user": "ask_user",
        },
    )
    graph.add_edge("secondary_retrieve", "retrieval_evaluator")
    graph.add_edge("ask_user", END)

    # Phase 4 generation + Self-RAG reflection (cyclic: regenerate)
    graph.add_edge("generate", "self_rag_reflect")
    graph.add_conditional_edges(
        "self_rag_reflect",
        make_reflect_router(settings),
        {"end": END, "regenerate": "generate"},
    )

    return graph.compile()


# Back-compat alias (earlier phases referred to the front-end graph).
build_frontend_graph = build_graph


def run_query(query: str, *, llm: LLMClient, settings: Settings, retriever=None, tree_store=None) -> dict:
    """Convenience: run one query through the graph and return final state."""
    app = build_graph(llm=llm, settings=settings, retriever=retriever, tree_store=tree_store)
    return app.invoke({"query": query})

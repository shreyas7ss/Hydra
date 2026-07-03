"""Phase 3 integration: hybrid coarse filter -> PageIndex tree search -> CRAG -> answer."""

from hydra.config import Settings
from hydra.graph import run_query
from hydra.llm import EchoLLM
from hydra.pageindex import flatten_to_documents
from hydra.pageindex.search import TreeStore
from hydra.retrieval import HybridRetriever
from hydra.retrieval.embeddings import HashingEmbedder
from hydra.retrieval.rerank import LexicalReranker
from hydra.sample_data import sample_documents, sample_tree


def _retrieval():
    llm = EchoLLM()
    tree = sample_tree(llm)
    store = TreeStore()
    store.add(tree)
    docs = sample_documents() + flatten_to_documents(tree)
    retriever = HybridRetriever.from_documents(
        docs, embedder=HashingEmbedder(dim=256), reranker=LexicalReranker(), settings=Settings()
    )
    return retriever, store


def _nodes(state):
    return [s["node"] for s in state["trace"]]


def test_complex_query_routes_to_pageindex_and_returns_intact_node():
    retriever, store = _retrieval()
    state = run_query(
        "what drove the change in operating margin, including the footnote?",
        llm=EchoLLM(), settings=Settings(), retriever=retriever, tree_store=store,
    )
    assert state["retrieval_strategy"] == "pageindex"
    assert "pageindex_tree_search" in _nodes(state)
    # the answer's grounding context is the intact Margins node (table + footnote)
    top = state["candidates"][0]
    assert "Footnote 1" in top["text"] and "21.5%" in top["text"]
    assert state["answer"]


def test_query_without_tree_stays_on_hybrid_chunks():
    retriever, store = _retrieval()
    # A complex query whose best match (the MSA termination clause) has no tree ->
    # the router keeps the hybrid chunks and skips PageIndex.
    state = run_query(
        "how can either party terminate the agreement for material breach?",
        llm=EchoLLM(), settings=Settings(), retriever=retriever, tree_store=store,
    )
    assert "hybrid_retrieve" in _nodes(state)
    assert state.get("retrieval_strategy") != "pageindex"
    assert "pageindex_tree_search" not in _nodes(state)

"""Retrieval nodes — the terminal nodes of the Phase 1 front-end, now Phase 2-capable.

Each is a factory bound to an optional ``HybridRetriever``:

* retriever attached  -> real retrieval (BM25 fast path / dense+BM25+RRF+rerank).
* retriever is None   -> Phase 1 stub behaviour (records the path, ends the graph),
  so the front-end still runs and Phase 1 tests stay valid.

Phase 3 adds the PageIndex tree-search branch off ``hybrid_retrieve``.
"""

from __future__ import annotations


def make_direct_lookup(retriever=None, *, top_k: int = 10):
    """Fast path for simple/exact-identifier queries (BM25/SQL)."""

    def direct_lookup(state: dict) -> dict:
        query = state["query"]
        search_queries = state.get("search_queries") or [query]
        if retriever is None:
            return {
                "retrieval_path": "bm25_sql_direct",
                "search_queries": search_queries,
                "trace": [{"node": "direct_lookup",
                           "detail": "stub: BM25/SQL fast path (no retriever attached)"}],
            }
        results = retriever.sparse_search(query, top_k)
        return {
            "retrieval_path": "bm25_sql_direct",
            "search_queries": search_queries,
            "candidates": [r.as_dict() for r in results],
            "trace": [{"node": "direct_lookup", "detail": f"BM25 fast path -> {len(results)} hits"}],
        }

    return direct_lookup


def make_hybrid_retrieve(retriever=None, *, top_k: int = 10):
    """Agentic path: dense(HNSW) + BM25 -> RRF(k=60) -> cross-encoder rerank."""

    def hybrid_retrieve(state: dict) -> dict:
        queries = state.get("search_queries") or [state["query"]]
        if retriever is None:
            return {
                "retrieval_path": "hybrid_dense_bm25_rrf_rerank",
                "trace": [{"node": "hybrid_retrieve",
                           "detail": f"stub: would retrieve for {len(queries)} query(ies) (no retriever attached)"}],
            }
        results = retriever.retrieve(
            queries,
            original_query=state["query"],
            hyde_doc=state.get("hyde_doc") or None,
            top_k=top_k,
        )
        return {
            "retrieval_path": "hybrid_dense_bm25_rrf_rerank",
            "candidates": [r.as_dict() for r in results],
            "trace": [{"node": "hybrid_retrieve",
                       "detail": f"RRF(k=60)+rerank over {len(queries)} query(ies) -> {len(results)} candidates"}],
        }

    return hybrid_retrieve

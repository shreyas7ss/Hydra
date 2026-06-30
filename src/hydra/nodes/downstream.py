"""Downstream retrieval stubs — the terminal nodes of the Phase 1 front-end.

These are intentionally hollow: they record which path the front-end selected and end
the graph. Phase 2 replaces ``hybrid_retrieve`` with dense(HNSW) + BM25 -> RRF(k=60) ->
cross-encoder reranking, and ``direct_lookup`` with the BM25/SQL fast path; Phase 3 adds
the PageIndex tree-search branch.
"""

from __future__ import annotations


def direct_lookup(state: dict) -> dict:
    """Fast path for simple/exact-identifier queries (BM25/SQL — stubbed)."""
    return {
        "retrieval_path": "bm25_sql_direct",
        "search_queries": state.get("search_queries") or [state["query"]],
        "trace": [{"node": "direct_lookup", "detail": "stub: BM25/SQL fast path (Phase 2)"}],
    }


def hybrid_retrieve(state: dict) -> dict:
    """Agentic path entry: dense + BM25 fusion + rerank over the fan-out (stubbed)."""
    n = len(state.get("search_queries", []))
    return {
        "retrieval_path": "hybrid_dense_bm25_rrf_rerank",
        "trace": [
            {"node": "hybrid_retrieve", "detail": f"stub: would retrieve for {n} query(ies) (Phase 2)"}
        ],
    }

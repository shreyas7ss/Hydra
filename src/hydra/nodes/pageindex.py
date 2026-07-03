"""Phase 3 node — fine-grained PageIndex tree search after the hybrid coarse filter.

The hybrid retriever has already narrowed to candidate nodes; this node identifies the
source document(s) of the top candidates, and if a PageTree exists for them, navigates the
tree to return the **intact** node (section + table + footnote) in place of the fragmented
chunks. The unchanged CRAG evaluator then grades the result.
"""

from __future__ import annotations

from hydra.config import Settings
from hydra.llm import LLMClient
from hydra.pageindex.search import TreeStore, tree_search


def _source_votes(candidates: list[dict], settings: Settings) -> list[str]:
    """All source docs ranked by rank-weighted votes over the top-k chunks — one
    lexically-lucky chunk can no longer hijack doc selection on its own."""
    votes: dict[str, float] = {}
    for rank, c in enumerate(candidates[: settings.pageindex_doc_vote_k]):
        src = (c.get("metadata") or {}).get("source")
        if src:
            votes[src] = votes.get(src, 0.0) + 1.0 / (rank + 1)
    return [src for src, _ in sorted(votes.items(), key=lambda x: x[1], reverse=True)]


def _vote_docs(candidates: list[dict], tree_store: TreeStore, settings: Settings) -> list[str]:
    """Tree-backed docs to navigate, but only when a tree-backed doc actually wins the
    overall vote — if a non-tree source dominates, PageIndex must not displace it."""
    ranked = _source_votes(candidates, settings)
    if not ranked or ranked[0] not in tree_store:
        return []
    return [src for src in ranked if src in tree_store][: settings.pageindex_top_docs]


def make_pageindex_search(tree_store: TreeStore, llm: LLMClient, settings: Settings):
    """Build the ``pageindex_tree_search`` node bound to a TreeStore + LLM."""

    def pageindex_tree_search(state: dict) -> dict:
        candidates = state.get("candidates", []) or []
        targets = _vote_docs(candidates, tree_store, settings)

        if not targets:
            return {
                "retrieval_strategy": "hybrid",
                "trace": [{"node": "pageindex_tree_search",
                           "detail": "no tree for top doc; kept hybrid chunks"}],
            }

        results = []
        paths: list[str] = []
        for src in targets:
            tree = tree_store.get(src)
            nodes, tree_paths = tree_search(tree, state["query"], llm, settings)
            results.extend(nodes)
            paths.extend(tree_paths)

        # +2 headroom so cross-referenced nodes appended by tree_search survive the cap.
        results = results[: settings.pageindex_max_nodes + 2]
        return {
            "candidates": [r.as_dict() for r in results],
            "retrieval_strategy": "pageindex",
            "trace": [{"node": "pageindex_tree_search",
                       "detail": f"navigated [{'; '.join(paths)}] -> {len(results)} intact node(s)"}],
        }

    return pageindex_tree_search


def make_retrieval_router(tree_store: TreeStore, settings: Settings):
    """Conditional-edge fn after hybrid_retrieve: pageindex | evaluate."""

    def route_retrieval(state: dict) -> str:
        if not settings.enable_pageindex or not tree_store:
            return "evaluate"
        candidates = state.get("candidates", []) or []
        if not candidates:
            return "evaluate"
        return "pageindex" if _vote_docs(candidates, tree_store, settings) else "evaluate"

    return route_retrieval

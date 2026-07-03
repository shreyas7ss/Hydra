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


def make_pageindex_search(tree_store: TreeStore, llm: LLMClient, settings: Settings):
    """Build the ``pageindex_tree_search`` node bound to a TreeStore + LLM."""

    def pageindex_tree_search(state: dict) -> dict:
        candidates = state.get("candidates", []) or []

        # Which of the top hybrid docs have a tree to navigate?
        targets: list[str] = []
        for c in candidates[: settings.pageindex_top_docs]:
            src = (c.get("metadata") or {}).get("source")
            if src and src in tree_store and src not in targets:
                targets.append(src)

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
            nodes, path = tree_search(tree, state["query"], llm, settings)
            results.extend(nodes)
            paths.append(" > ".join(path))

        results = results[: settings.pageindex_max_nodes]
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
        top_source = (candidates[0].get("metadata") or {}).get("source")
        return "pageindex" if top_source in tree_store else "evaluate"

    return route_retrieval

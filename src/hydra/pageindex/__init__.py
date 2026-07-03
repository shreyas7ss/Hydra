"""Phase 3 — Vectorless / PageIndex.

A hierarchical document tree ("reasoning over searching"): the LLM navigates
Root -> Section -> Page to locate data and returns **intact nodes**, so a table is never
severed from the footnote that modifies it. Used as the fine-grained stage *after* the
hybrid coarse filter narrows to the right document.
"""

from hydra.pageindex.build import (
    build_tree_from_pdf,
    build_tree_from_sections,
    build_tree_from_text,
)
from hydra.pageindex.search import TreeStore, tree_search
from hydra.pageindex.tree import PageNode, PageTree, flatten_to_documents, full_text

__all__ = [
    "PageNode",
    "PageTree",
    "full_text",
    "flatten_to_documents",
    "build_tree_from_sections",
    "build_tree_from_text",
    "build_tree_from_pdf",
    "TreeStore",
    "tree_search",
]

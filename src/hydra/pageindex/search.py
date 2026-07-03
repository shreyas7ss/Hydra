"""Reasoning-based tree search + the TreeStore.

The LLM walks Root -> Section -> Page using node summaries (reasoning over searching),
then the selected node is returned *intact* (node + full subtree) so structure is never
fragmented. The navigation path is returned for the audit trail.
"""

from __future__ import annotations

from hydra.config import Settings
from hydra.llm import LLMClient, parse_json
from hydra.pageindex.tree import PageNode, PageTree, full_text
from hydra.retrieval.documents import Document, ScoredDoc

NAV_SYSTEM = """You are navigating a document tree to find where a question is answered.
Given the question and the current node's child sections (each with an index, title, and
summary), choose the single child index most likely to contain the answer, or stop if the
current node itself is already the right place.

Respond with ONLY JSON: {"choice": <child index or -1 to stop here>, "reason": "<short>"}
TASK: tree_nav"""


def _subtree_keywords(node: PageNode, limit: int = 80) -> str:
    """Distinctive content words found anywhere under a node — signals what the branch
    contains beyond its one-line summary (helps navigate to the right deep section)."""
    from hydra.retrieval.text import tokenize

    seen: list[str] = []
    known: set[str] = set()
    for tok in tokenize(full_text(node)):
        if tok not in known:
            known.add(tok)
            seen.append(tok)
        if len(seen) >= limit:
            break
    return " ".join(seen)


def _format_children(children: list[PageNode]) -> str:
    lines = []
    for i, child in enumerate(children):
        summary = child.summary or child.title
        lines.append(f"{i}. {child.title} :: {summary} :: keys: {_subtree_keywords(child)}")
    return "\n".join(lines)


def _navigate(node: PageNode, query: str, llm: LLMClient) -> int:
    """Return the chosen child index, or -1 to stop at the current node."""
    user = f"Question: {query}\n\nChildren:\n{_format_children(node.children)}"
    data = parse_json(llm.complete(system=NAV_SYSTEM, user=user)) or {}
    try:
        choice = int(data.get("choice", -1))
    except (TypeError, ValueError):
        choice = -1
    if 0 <= choice < len(node.children):
        return choice
    return -1


def tree_search(
    tree: PageTree,
    query: str,
    llm: LLMClient,
    settings: Settings,
) -> tuple[list[ScoredDoc], list[str]]:
    """Descend the tree for ``query``; return the intact node(s) + the navigation path."""
    node = tree.root
    path = [node.title]
    depth = 0
    while node.children and depth < settings.pageindex_max_depth:
        choice = _navigate(node, query, llm)
        if choice < 0:
            break
        node = node.children[choice]
        path.append(node.title)
        depth += 1

    section = " / ".join(node.section_path) or node.title
    document = Document(
        id=node.id,
        text=full_text(node),  # intact: node + entire subtree
        metadata={
            "source": tree.source,
            "doc_id": tree.doc_id,
            "node_id": node.id,
            "page": node.page,
            "section": section,
        },
    )
    scored = ScoredDoc(document, 1.0, [f"pageindex:{' > '.join(path)}"])
    return [scored], path


class TreeStore:
    """Registry of PageTrees, keyed by both ``source`` and ``doc_id`` for lookup from
    hybrid-candidate metadata."""

    def __init__(self) -> None:
        self._trees: dict[str, PageTree] = {}

    def add(self, tree: PageTree) -> None:
        self._trees[tree.source] = tree
        self._trees[tree.doc_id] = tree

    def get(self, key: str) -> PageTree | None:
        return self._trees.get(key)

    def __contains__(self, key: object) -> bool:
        return key in self._trees

    def __bool__(self) -> bool:
        return bool(self._trees)

    def __len__(self) -> int:
        return len(set(id(t) for t in self._trees.values()))

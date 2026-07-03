"""Reasoning-based tree search + the TreeStore.

The LLM walks Root -> Section -> Page using node summaries (reasoning over searching),
then the selected node is returned *intact* (node + full subtree) so structure is never
fragmented. The navigation path is returned for the audit trail.
"""

from __future__ import annotations

import re

from hydra.config import Settings
from hydra.llm import LLMClient, parse_json
from hydra.pageindex.tree import PageNode, PageTree, full_text
from hydra.retrieval.documents import Document, ScoredDoc

NAV_SYSTEM = """You are navigating a document tree to find where a question is answered.
Given the question and the current node's child sections (each with an index, title, and
summary), rank the child indices most likely to contain the answer (best first). If the
current node itself is already the right place, return an empty list.

Respond with ONLY JSON: {"choices": [<child indices, best first>], "reason": "<short>"}
TASK: tree_nav"""

# Cross-references like "see Note 7" / "refer to Item 7A" — endemic in SEC filings.
_XREF = re.compile(r"\b(?:see|refer to)\s+(note\s+\d+[a-z]?|item\s+\d+[a-z]?)", re.IGNORECASE)


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


def _navigate(node: PageNode, query: str, llm: LLMClient) -> list[int]:
    """Return ranked child indices (best first); empty list = stop at this node."""
    user = f"Question: {query}\n\nChildren:\n{_format_children(node.children)}"
    data = parse_json(llm.complete(system=NAV_SYSTEM, user=user)) or {}
    raw = data.get("choices", data.get("choice", []))  # tolerate old single-choice form
    if isinstance(raw, (int, float)):
        raw = [raw]
    choices: list[int] = []
    for item in raw if isinstance(raw, list) else []:
        try:
            idx = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(node.children) and idx not in choices:
            choices.append(idx)
    return choices


def _find_by_title(tree: PageTree, ref: str) -> PageNode | None:
    ref_low = " ".join(ref.lower().split())
    for node in tree.iter_nodes():
        if ref_low in " ".join(node.title.lower().split()):
            return node
    return None


def _as_scored(tree: PageTree, node: PageNode, path: str, score: float) -> ScoredDoc:
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
    return ScoredDoc(document, score, [f"pageindex:{path}"])


def tree_search(
    tree: PageTree,
    query: str,
    llm: LLMClient,
    settings: Settings,
) -> tuple[list[ScoredDoc], list[str]]:
    """Beam-search the tree for ``query``; return intact node(s) + navigation paths.

    Beam width > 1 hedges against a wrong turn at a high level; multi-node returns
    cover questions whose evidence spans sections; "see Note X" cross-references in
    selected nodes pull the referenced node in as well.
    """
    beam = max(1, settings.pageindex_beam_width)
    frontier: list[tuple[PageNode, list[str]]] = [(tree.root, [tree.root.title])]
    landed: list[tuple[PageNode, list[str]]] = []

    for _depth in range(settings.pageindex_max_depth):
        next_frontier: list[tuple[PageNode, list[str]]] = []
        for node, path in frontier:
            if not node.children:
                landed.append((node, path))
                continue
            choices = _navigate(node, query, llm)
            if not choices:
                landed.append((node, path))
                continue
            for idx in choices[:beam]:
                child = node.children[idx]
                next_frontier.append((child, path + [child.title]))
        frontier = next_frontier[:beam]  # LLM preference order; keep the beam bounded
        if not frontier:
            break
    landed.extend(frontier)

    # Dedup by node id, preserving arrival (preference) order.
    seen: set[str] = set()
    selected: list[tuple[PageNode, list[str]]] = []
    for node, path in landed:
        if node.id not in seen:
            seen.add(node.id)
            selected.append((node, path))
    selected = selected[: settings.pageindex_max_nodes]

    results: list[ScoredDoc] = []
    paths: list[str] = []
    for rank, (node, path) in enumerate(selected):
        path_str = " > ".join(path)
        results.append(_as_scored(tree, node, path_str, 1.0 - 0.1 * rank))
        paths.append(path_str)

    # Follow "see Note X" cross-references out of the selected nodes.
    if settings.pageindex_follow_refs:
        for node, _path in list(selected):
            for ref in _XREF.findall(full_text(node)):
                target = _find_by_title(tree, ref)
                if target and target.id not in seen and len(results) < settings.pageindex_max_nodes + 2:
                    seen.add(target.id)
                    results.append(_as_scored(tree, target, f"xref:{ref}", 0.5))
                    paths.append(f"xref -> {target.title}")

    return results, paths


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

"""PageIndex tree data structures.

A ``PageTree`` is the machine-readable map of one document's organization. Each ``PageNode``
carries its physical page and section path (metadata attachment) plus an LLM summary used to
guide traversal. ``full_text`` returns a node together with its whole subtree, in order — this
is the intact-context guarantee that keeps footnotes attached to their tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from hydra.retrieval.documents import Document


@dataclass
class PageNode:
    id: str
    title: str
    content: str = ""
    page: int | None = None
    section_path: list[str] = field(default_factory=list)
    summary: str = ""
    children: list["PageNode"] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return not self.children


class PageTree:
    def __init__(self, doc_id: str, root: PageNode, source: str | None = None) -> None:
        self.doc_id = doc_id
        self.root = root
        self.source = source or doc_id

    def iter_nodes(self) -> Iterator[PageNode]:
        stack = [self.root]
        while stack:
            node = stack.pop()
            yield node
            # push children reversed so iteration is left-to-right
            stack.extend(reversed(node.children))

    def find(self, node_id: str) -> PageNode | None:
        for node in self.iter_nodes():
            if node.id == node_id:
                return node
        return None


def full_text(node: PageNode) -> str:
    """Node title + content + every descendant, in document order (intact context)."""
    parts: list[str] = []
    header = node.title.strip()
    body = node.content.strip()
    if header:
        parts.append(header)
    if body:
        parts.append(body)
    for child in node.children:
        parts.append(full_text(child))
    return "\n".join(p for p in parts if p)


def flatten_to_documents(tree: PageTree) -> list[Document]:
    """One Document per node that has its own content — the corpus the hybrid coarse
    filter indexes. Metadata points back to the tree location so a hit maps to a node."""
    docs: list[Document] = []
    for node in tree.iter_nodes():
        if not node.content.strip():
            continue  # structural-only node; nothing to index on its own
        docs.append(
            Document(
                id=node.id,
                text=f"{node.title}\n{node.content}".strip(),
                metadata={
                    "source": tree.source,
                    "doc_id": tree.doc_id,
                    "node_id": node.id,
                    "page": node.page,
                    "section": " / ".join(node.section_path) or node.title,
                },
            )
        )
    return docs

"""Tree construction + the TOC fallback cascade (self-correction).

Stages: hierarchical parse -> JSON tree -> metadata attachment -> per-node LLM summaries.

Fallback cascade for messy inputs:
  Mode 1: structured sections *with* physical page numbers.
  Mode 2: structured sections *without* page numbers.
  Mode 3: pure-LLM segmentation of unstructured text.
"""

from __future__ import annotations

from typing import Any

from hydra.llm import LLMClient, parse_json
from hydra.pageindex.tree import PageNode, PageTree

SUMMARY_SYSTEM = """Summarize the following document section in one concise sentence that
captures what a reader would find here (for navigation). Return only the sentence.
TASK: node_summary"""

SEGMENT_SYSTEM = """Split the document text into a sequence of titled sections following its
natural structure. Return ONLY a JSON array of objects: [{"title": "...", "content": "..."}].
TASK: segment"""


def _as_section_dict(section: Any) -> dict:
    """Accept either a parse.Section dataclass or a plain dict."""
    if isinstance(section, dict):
        return {
            "title": section.get("title", ""),
            "content": section.get("content", ""),
            "page": section.get("page"),
            "level": section.get("level", 1),
        }
    return {
        "title": getattr(section, "title", ""),
        "content": getattr(section, "content", ""),
        "page": getattr(section, "page", None),
        "level": getattr(section, "level", 1),
    }


def _summarize(node: PageNode, llm: LLMClient) -> str:
    basis = node.content.strip() or node.title.strip()
    if not basis:
        return ""
    return llm.complete(system=SUMMARY_SYSTEM, user=basis[:1500]).strip()


def build_tree_from_sections(
    doc_id: str,
    sections: list[Any],
    llm: LLMClient,
    *,
    source: str | None = None,
) -> PageTree:
    """Build a tree from ordered sections (Mode 1/2) and attach LLM summaries."""
    root = PageNode(id=f"{doc_id}::root", title=doc_id, section_path=[])
    stack: list[tuple[int, PageNode]] = [(0, root)]

    for i, raw in enumerate(sections):
        sec = _as_section_dict(raw)
        level = max(1, int(sec["level"]))
        node = PageNode(
            id=f"{doc_id}::n{i}",
            title=sec["title"],
            content=sec["content"].strip(),
            page=sec["page"],
        )
        # Pop to the nearest ancestor with a strictly smaller level.
        while len(stack) > 1 and stack[-1][0] >= level:
            stack.pop()
        parent = stack[-1][1]
        node.section_path = parent.section_path + [node.title]
        parent.children.append(node)
        stack.append((level, node))

    tree = PageTree(doc_id, root, source=source)
    for node in tree.iter_nodes():
        if node is not root:
            node.summary = _summarize(node, llm)
    return tree


def build_tree_from_text(doc_id: str, text: str, llm: LLMClient, *, source: str | None = None) -> PageTree:
    """Mode 3: no structure available — let the LLM segment the raw text."""
    data = parse_json(llm.complete(system=SEGMENT_SYSTEM, user=text[:6000]))
    sections: list[dict] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and item.get("title"):
                sections.append({"title": item["title"], "content": item.get("content", ""), "level": 1})
    if not sections:
        # Last-resort: one flat node so we always return a usable tree.
        sections = [{"title": doc_id, "content": text, "level": 1}]
    return build_tree_from_sections(doc_id, sections, llm, source=source)


def build_tree_from_pdf(path: str, llm: LLMClient, *, doc_id: str | None = None, source: str | None = None) -> PageTree:
    """Ingest a real PDF through the fallback cascade."""
    import os

    from hydra.pageindex.parse import parse_pdf

    name = os.path.basename(path) or path
    doc_id = doc_id or name
    source = source or name
    sections = parse_pdf(path)

    has_headings = any(s.level and s.title != "(preamble)" for s in sections)
    if sections and has_headings:
        # Mode 1 if page numbers present, else Mode 2 (same builder; page may be None).
        return build_tree_from_sections(doc_id, sections, llm, source=source)

    # Mode 3: nothing structured recovered -> pure-LLM segmentation over the raw text.
    raw = "\n".join(s.content for s in sections) if sections else ""
    return build_tree_from_text(doc_id, raw, llm, source=source)

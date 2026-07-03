"""Hierarchical PDF parsing (stage 1 of tree construction).

Uses ``pdfplumber`` per-line font metrics — not just text shape — to infer the heading
hierarchy:

* A document-wide **body size** is estimated (median char size); lines set meaningfully
  larger are headings, levelled by size rank.
* Numbered headings ("1.2 Liquidity") get their level from numbering depth.
* **Running headers/footers** (the same line repeating across many pages) are dropped
  before they can pollute sections.
* Tables are attached to the section in progress, keeping values with their headers.

PDF structure inference is still heuristic (PDFs carry no reliable outline); the caller's
fallback cascade (build.py) downgrades to pure-LLM segmentation when this yields nothing.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from statistics import median

# Numbered headings like "1", "1.2", "3.4.1", or SEC-style "Item 7." / "ITEM 7A."
_NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(\S.*)$")
_SEC_ITEM = re.compile(r"^item\s+(\d+[a-z]?)\.?\s*(.*)$", re.IGNORECASE)

_HEADING_SIZE_RATIO = 1.12   # line size vs body size to count as a heading
_RUNNING_LINE_FRACTION = 0.4  # line repeating on >40% of pages = header/footer


@dataclass
class Section:
    title: str
    content: str = ""
    page: int | None = None
    level: int = 1
    tables: list[str] = field(default_factory=list)


@dataclass
class _Line:
    text: str
    size: float
    page: int


def _text_heading_level(line: str) -> int | None:
    """Text-shape fallback signals (used when font metrics are inconclusive)."""
    stripped = line.strip()
    if not stripped or len(stripped) > 90:
        return None
    m = _NUMBERED.match(stripped)
    if m:
        return stripped.count(".", 0, len(m.group(1))) + 1
    if _SEC_ITEM.match(stripped):
        return 1
    words = stripped.split()
    if len(words) <= 8 and not stripped.endswith((".", ",", ";", ":")):
        letters = [c for c in stripped if c.isalpha()]
        if letters and stripped.upper() == stripped:
            return 1
    return None


def _extract_lines(pdf) -> list[_Line]:
    lines: list[_Line] = []
    for page_no, page in enumerate(pdf.pages, start=1):
        for raw in page.extract_text_lines() or []:
            text = (raw.get("text") or "").strip()
            if not text:
                continue
            chars = raw.get("chars") or []
            sizes = [c.get("size", 0.0) for c in chars if c.get("size")]
            size = median(sizes) if sizes else 0.0
            lines.append(_Line(text=text, size=size, page=page_no))
    return lines


def _running_lines(lines: list[_Line], n_pages: int) -> set[str]:
    """Lines repeating across many pages are headers/footers/page numbers."""
    if n_pages < 3:
        return set()
    counts = Counter(ln.text for ln in lines)
    threshold = max(3, int(n_pages * _RUNNING_LINE_FRACTION))
    running = {text for text, c in counts.items() if c >= threshold}
    # Bare page numbers ("42", "Page 42 of 118")
    running |= {ln.text for ln in lines
                if re.fullmatch(r"(page\s+)?\d+(\s+of\s+\d+)?", ln.text, re.IGNORECASE)}
    return running


def _size_levels(lines: list[_Line], body_size: float) -> dict[float, int]:
    """Map distinct heading font sizes to levels (largest = 1)."""
    heading_sizes = sorted(
        {round(ln.size, 1) for ln in lines
         if ln.size >= body_size * _HEADING_SIZE_RATIO and len(ln.text) <= 90},
        reverse=True,
    )
    return {size: i + 1 for i, size in enumerate(heading_sizes[:4])}


def parse_pdf(path: str) -> list[Section]:
    """Parse a PDF into an ordered list of Sections with page numbers + tables."""
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        lines = _extract_lines(pdf)
        n_pages = len(pdf.pages)
        skip = _running_lines(lines, n_pages)

        sizes = [ln.size for ln in lines if ln.size > 0]
        body_size = median(sizes) if sizes else 0.0
        size_level = _size_levels(lines, body_size) if body_size else {}

        sections: list[Section] = []
        current: Section | None = None
        lines_by_page: dict[int, list[_Line]] = {}
        for ln in lines:
            lines_by_page.setdefault(ln.page, []).append(ln)

        for page_no, page in enumerate(pdf.pages, start=1):
            for ln in lines_by_page.get(page_no, []):
                if ln.text in skip:
                    continue
                # Font-size signal first; text-shape signals as fallback.
                level = size_level.get(round(ln.size, 1)) if ln.size else None
                if level is None:
                    level = _text_heading_level(ln.text)
                elif _NUMBERED.match(ln.text):
                    # Numbering depth beats size rank for numbered headings.
                    level = _text_heading_level(ln.text) or level

                if level is not None and len(ln.text) <= 90:
                    current = Section(title=ln.text, page=page_no, level=level)
                    sections.append(current)
                else:
                    if current is None:
                        current = Section(title="(preamble)", page=page_no, level=1)
                        sections.append(current)
                    current.content += ln.text + "\n"

            for table in page.extract_tables() or []:
                table_text = _table_to_text(table)
                if not table_text:
                    continue
                if current is None:
                    current = Section(title="(preamble)", page=page_no, level=1)
                    sections.append(current)
                current.tables.append(table_text)
                current.content += "\n" + table_text + "\n"

    return sections


def _table_to_text(table: list[list]) -> str:
    rows = []
    for row in table:
        cells = [("" if c is None else str(c)).strip() for c in row]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)

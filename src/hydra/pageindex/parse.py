"""Hierarchical PDF parsing (stage 1 of tree construction).

Uses ``pdfplumber`` to pull text + tables per page and infer a heading hierarchy. PDF
structure inference is inherently heuristic (PDFs carry no reliable outline), so the
caller's fallback cascade (build.py) downgrades to page-number-less or pure-LLM
segmentation when this yields nothing useful.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Numbered headings like "1", "1.2", "3.4.1" followed by a title.
_NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(\S.*)$")


@dataclass
class Section:
    title: str
    content: str = ""
    page: int | None = None
    level: int = 1
    tables: list[str] = field(default_factory=list)


def _heading_level(line: str) -> int | None:
    """Return a heading level (1 = top) if the line looks like a heading, else None."""
    stripped = line.strip()
    if not stripped or len(stripped) > 90:
        return None
    m = _NUMBERED.match(stripped)
    if m:
        return stripped.count(".", 0, len(m.group(1))) + 1  # depth from the number
    words = stripped.split()
    # ALL-CAPS or short Title-Case lines with no terminal punctuation read as headings.
    if len(words) <= 8 and not stripped.endswith((".", ",", ";", ":")):
        letters = [c for c in stripped if c.isalpha()]
        if letters and stripped.upper() == stripped:
            return 1
        if all(w[:1].isupper() for w in words if w[:1].isalpha()):
            return 2
    return None


def _table_to_text(table: list[list]) -> str:
    rows = []
    for row in table:
        cells = [("" if c is None else str(c)).strip() for c in row]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def parse_pdf(path: str) -> list[Section]:
    """Parse a PDF into an ordered list of Sections with page numbers + tables."""
    import pdfplumber

    sections: list[Section] = []
    current: Section | None = None

    with pdfplumber.open(path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            for raw_line in text.split("\n"):
                line = raw_line.rstrip()
                if not line.strip():
                    continue
                level = _heading_level(line)
                if level is not None:
                    current = Section(title=line.strip(), page=page_no, level=level)
                    sections.append(current)
                else:
                    if current is None:
                        current = Section(title="(preamble)", page=page_no, level=1)
                        sections.append(current)
                    current.content += (line + "\n")
            # Attach tables to the current section so a table stays with its section.
            for table in page.extract_tables() or []:
                table_text = _table_to_text(table)
                if not table_text:
                    continue
                if current is None:
                    current = Section(title="(preamble)", page=page_no, level=1)
                    sections.append(current)
                current.tables.append(table_text)
                current.content += ("\n" + table_text + "\n")

    return sections

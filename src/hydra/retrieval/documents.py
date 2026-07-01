"""Document + scored-result types.

``metadata`` carries page / section / source so the downstream audit trail and (Phase 4)
citations can name exactly where an answer was grounded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredDoc:
    document: Document
    score: float
    # Which retriever/query lists surfaced this doc — kept for auditability.
    sources: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.document.id,
            "text": self.document.text,
            "metadata": self.document.metadata,
            "score": round(float(self.score), 6),
            "sources": self.sources,
        }

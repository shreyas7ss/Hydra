"""Precision reranking — the single largest precision gain in the stack.

* ``CrossEncoderReranker`` — the directive-mandated cross-encoder (jointly encodes
  query+doc). Needs the `rerank` extra (sentence-transformers/torch); lazy import.
* ``LexicalReranker``      — offline default. Token-overlap scoring: not a real
  cross-encoder, but keeps the pipeline runnable with no model download. Production
  should set HYDRA_RERANKER_PROVIDER=cross-encoder.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from hydra.retrieval.documents import Document
from hydra.retrieval.text import tokenize


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, documents: list[Document]) -> list[tuple[Document, float]]: ...


class LexicalReranker:
    """Score by query/doc token overlap (recall-weighted). Deterministic, offline."""

    def rerank(self, query: str, documents: list[Document]) -> list[tuple[Document, float]]:
        q_terms = set(tokenize(query))
        scored: list[tuple[Document, float]] = []
        for doc in documents:
            d_terms = set(tokenize(doc.text))
            if not q_terms:
                score = 0.0
            else:
                overlap = len(q_terms & d_terms)
                # Reward covering the query terms; lightly penalise nothing else.
                score = overlap / len(q_terms)
            scored.append((doc, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - optional heavy extra
            raise RuntimeError(
                "The 'rerank' extra is not installed. Run `uv sync --extra rerank`, "
                "or use the offline lexical reranker (HYDRA_RERANKER_PROVIDER=lexical)."
            ) from exc
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: list[Document]) -> list[tuple[Document, float]]:
        if not documents:
            return []
        pairs = [(query, doc.text) for doc in documents]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(documents, scores), key=lambda x: float(x[1]), reverse=True)
        return [(doc, float(score)) for doc, score in ranked]


def build_reranker(settings, *, demo: bool = False) -> Reranker:
    provider = "lexical" if demo else settings.reranker_provider.lower()
    if provider == "lexical":
        return LexicalReranker()
    if provider in ("cross-encoder", "cross_encoder", "crossencoder"):
        return CrossEncoderReranker(settings.reranker_model)
    raise RuntimeError(f"Unknown reranker provider: {settings.reranker_provider!r}")

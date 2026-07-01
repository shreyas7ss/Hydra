"""Phase 2 — Coarse Retrieval & Precision Fusion.

dense(HNSW) + BM25(sparse)  ->  Reciprocal Rank Fusion (k=60)  ->  cross-encoder rerank.

The public surface is ``HybridRetriever``; everything else is a swappable component
behind a protocol (embedder, reranker) so production models drop in via config.
"""

from hydra.retrieval.documents import Document, ScoredDoc
from hydra.retrieval.embeddings import Embedder, HashingEmbedder, build_embedder
from hydra.retrieval.pipeline import HybridRetriever
from hydra.retrieval.rerank import Reranker, build_reranker

__all__ = [
    "Document",
    "ScoredDoc",
    "Embedder",
    "HashingEmbedder",
    "build_embedder",
    "Reranker",
    "build_reranker",
    "HybridRetriever",
]

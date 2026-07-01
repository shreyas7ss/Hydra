"""HybridRetriever — the Phase 2 two-stage pipeline.

Per query in the (Phase 1) fan-out we run dense + BM25; the HyDE doc seeds an extra
dense search (answer-to-answer similarity). All ranked lists are fused with RRF(k=60),
the top ``rerank_pool`` are reranked against the *original* user query, and the top
``retrieval_top_k`` are returned.
"""

from __future__ import annotations

from hydra.config import Settings
from hydra.retrieval.dense import DenseIndex
from hydra.retrieval.documents import Document, ScoredDoc
from hydra.retrieval.embeddings import Embedder, build_embedder
from hydra.retrieval.fusion import reciprocal_rank_fusion
from hydra.retrieval.rerank import Reranker, build_reranker
from hydra.retrieval.sparse import BM25Index


def _short(text: str, n: int = 24) -> str:
    text = text.strip().replace("\n", " ")
    return text if len(text) <= n else text[: n - 1] + "..."


class HybridRetriever:
    def __init__(
        self,
        *,
        embedder: Embedder,
        reranker: Reranker,
        settings: Settings,
        enable_dense: bool | None = None,
        enable_sparse: bool | None = None,
    ) -> None:
        self.settings = settings
        self.reranker = reranker
        self._enable_dense = settings.enable_dense if enable_dense is None else enable_dense
        self._enable_sparse = settings.enable_sparse if enable_sparse is None else enable_sparse
        if not (self._enable_dense or self._enable_sparse):
            raise ValueError("At least one of dense/sparse retrieval must be enabled.")
        self.dense = DenseIndex(embedder) if self._enable_dense else None
        self.sparse = BM25Index() if self._enable_sparse else None
        self._docs: dict[str, Document] = {}

    # --- construction helpers ------------------------------------------------
    @classmethod
    def from_documents(cls, documents: list[Document], **kwargs) -> "HybridRetriever":
        retriever = cls(**kwargs)
        retriever.index(documents)
        return retriever

    @classmethod
    def from_settings(cls, documents, *, settings, demo: bool = False) -> "HybridRetriever":
        return cls.from_documents(
            documents,
            embedder=build_embedder(settings, demo=demo),
            reranker=build_reranker(settings, demo=demo),
            settings=settings,
        )

    def index(self, documents: list[Document]) -> None:
        self._docs = {d.id: d for d in documents}
        if self.dense:
            self.dense.build(documents)
        if self.sparse:
            self.sparse.build(documents)

    # --- retrieval -----------------------------------------------------------
    def sparse_search(self, query: str, top_k: int | None = None) -> list[ScoredDoc]:
        """BM25-only fast path for the direct-lookup route."""
        if not self.sparse:
            return []
        top_k = top_k or self.settings.retrieval_top_k
        return [
            ScoredDoc(self._docs[doc_id], score, ["bm25"])
            for doc_id, score in self.sparse.search(query, top_k)
        ]

    def retrieve(
        self,
        queries: list[str],
        *,
        original_query: str,
        hyde_doc: str | None = None,
        top_k: int | None = None,
    ) -> list[ScoredDoc]:
        top_k = top_k or self.settings.retrieval_top_k
        per_q = self.settings.per_query_top_k

        labeled: list[tuple[str, list[tuple[str, float]]]] = []
        for q in queries:
            if self.dense:
                labeled.append((f"dense:{_short(q)}", self.dense.search(q, per_q)))
            if self.sparse:
                labeled.append((f"bm25:{_short(q)}", self.sparse.search(q, per_q)))
        if hyde_doc and self.dense:
            labeled.append(("dense:hyde", self.dense.search(hyde_doc, per_q)))

        fused, contributors = reciprocal_rank_fusion(labeled, k=self.settings.fusion_k)
        pool = [doc_id for doc_id, _ in fused[: self.settings.rerank_pool]]
        pool_docs = [self._docs[doc_id] for doc_id in pool]

        reranked = self.reranker.rerank(original_query, pool_docs)
        results: list[ScoredDoc] = []
        for doc, score in reranked[:top_k]:
            results.append(ScoredDoc(doc, float(score), contributors.get(doc.id, [])))
        return results

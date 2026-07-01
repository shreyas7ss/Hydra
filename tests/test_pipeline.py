import pytest

from hydra.config import Settings
from hydra.retrieval import HybridRetriever
from hydra.retrieval.documents import Document
from hydra.retrieval.embeddings import HashingEmbedder
from hydra.retrieval.rerank import LexicalReranker

DOCS = [
    Document(id="margin", text="operating margin for 2023 expanded to 21.5 percent from 18.2 percent in 2022",
             metadata={"page": 43}),
    Document(id="revenue", text="fiscal 2023 total net revenue was 1.24 billion dollars",
             metadata={"page": 42}),
    Document(id="clause", text="clause 7.2 limitation of liability caps aggregate damages",
             metadata={"page": 11}),
    Document(id="pto", text="paid time off accrues at 1.5 days per month up to 30 days",
             metadata={"page": 7}),
]


def _retriever(**overrides):
    settings = Settings(**overrides)
    return HybridRetriever.from_documents(
        DOCS,
        embedder=HashingEmbedder(dim=128),
        reranker=LexicalReranker(),
        settings=settings,
    )


def test_hybrid_retrieve_surfaces_relevant_doc_first():
    r = _retriever()
    results = r.retrieve(
        ["what was the operating margin in 2023?", "2023 operating margin"],
        original_query="what was the operating margin in 2023?",
        top_k=3,
    )
    assert results
    assert results[0].document.id == "margin"
    # audit: the top hit records which retriever lists surfaced it.
    assert results[0].sources


def test_sparse_search_fast_path():
    r = _retriever()
    results = r.sparse_search("limitation of liability clause", top_k=2)
    assert results[0].document.id == "clause"
    assert results[0].sources == ["bm25"]


def test_top_k_is_respected():
    r = _retriever()
    results = r.retrieve(["revenue"], original_query="revenue", top_k=2)
    assert len(results) <= 2


def test_hyde_doc_participates_in_fusion():
    r = _retriever()
    results = r.retrieve(
        ["margin"],
        original_query="margin",
        hyde_doc="operating margin for 2023 expanded to 21.5 percent",
        top_k=4,
    )
    ids = [r.document.id for r in results]
    assert "margin" in ids


def test_requires_at_least_one_retriever():
    with pytest.raises(ValueError):
        HybridRetriever(
            embedder=HashingEmbedder(dim=16),
            reranker=LexicalReranker(),
            settings=Settings(enable_dense=False, enable_sparse=False),
        )


def test_dense_only_and_sparse_only_both_work():
    dense_only = _retriever(enable_sparse=False)
    sparse_only = _retriever(enable_dense=False)
    for r in (dense_only, sparse_only):
        results = r.retrieve(["operating margin 2023"], original_query="operating margin 2023", top_k=3)
        assert results

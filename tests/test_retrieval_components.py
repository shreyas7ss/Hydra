from hydra.retrieval.documents import Document
from hydra.retrieval.embeddings import HashingEmbedder
from hydra.retrieval.fusion import reciprocal_rank_fusion
from hydra.retrieval.rerank import LexicalReranker
from hydra.retrieval.sparse import BM25Index

DOCS = [
    Document(id="d1", text="operating margin expanded in 2023 to 21.5 percent"),
    Document(id="d2", text="clause 7.2 limitation of liability caps aggregate damages"),
    Document(id="d3", text="paid time off accrues at 1.5 days per month"),
]


# --- BM25 ---
def test_bm25_ranks_term_match_first():
    idx = BM25Index()
    idx.build(DOCS)
    results = idx.search("limitation of liability clause", top_k=3)
    assert results[0][0] == "d2"


def test_bm25_empty_query_returns_nothing():
    idx = BM25Index()
    idx.build(DOCS)
    assert idx.search("xyzzy", top_k=3) == []


# --- Hashing embedder ---
def test_hashing_embedder_is_deterministic_and_normalised():
    emb = HashingEmbedder(dim=64)
    a = emb.embed(["operating margin 2023"])[0]
    b = emb.embed(["operating margin 2023"])[0]
    assert a == b
    assert abs(sum(x * x for x in a) - 1.0) < 1e-9
    assert len(a) == 64


def test_hashing_embedder_shared_tokens_are_more_similar():
    emb = HashingEmbedder(dim=128)
    q, near, far = emb.embed(
        ["operating margin 2023", "operating margin improved 2023", "paid time off policy"]
    )
    dot_near = sum(x * y for x, y in zip(q, near))
    dot_far = sum(x * y for x, y in zip(q, far))
    assert dot_near > dot_far


# --- RRF ---
def test_rrf_rewards_agreement_across_lists():
    lists = [
        ("dense", [("d1", 0.9), ("d2", 0.8)]),
        ("bm25", [("d2", 5.0), ("d3", 1.0)]),
    ]
    fused, contributors = reciprocal_rank_fusion(lists, k=60)
    ranked_ids = [doc_id for doc_id, _ in fused]
    # d2 appears in both lists -> should rank first.
    assert ranked_ids[0] == "d2"
    assert set(contributors["d2"]) == {"dense", "bm25"}


def test_rrf_uses_rank_not_score_scale():
    # BM25 score of 100 must not dominate; only rank position matters.
    lists = [("a", [("x", 100.0), ("y", 99.0)]), ("b", [("y", 0.01), ("x", 0.005)])]
    fused, _ = reciprocal_rank_fusion(lists, k=60)
    # x is rank0 in a + rank1 in b; y is rank1 in a + rank0 in b -> tie, both present.
    assert {doc_id for doc_id, _ in fused} == {"x", "y"}


# --- Reranker ---
def test_lexical_reranker_orders_by_overlap():
    reranker = LexicalReranker()
    ranked = reranker.rerank("limitation of liability", DOCS)
    assert ranked[0][0].id == "d2"

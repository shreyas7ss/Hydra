from hydra.config import Settings
from hydra.llm import EchoLLM
from hydra.nodes.pageindex import _vote_docs, make_retrieval_router
from hydra.pageindex.build import build_tree_from_sections
from hydra.pageindex.search import TreeStore


def _store_with(doc_id: str) -> TreeStore:
    store = TreeStore()
    tree = build_tree_from_sections(
        doc_id, [{"title": "S", "content": "c", "level": 1}], EchoLLM(), source=doc_id
    )
    store.add(tree)
    return store


def _cand(src: str) -> dict:
    return {"id": src, "text": "x", "metadata": {"source": src}}


def test_majority_votes_beat_single_top_chunk():
    # Top-1 chunk is from no-tree doc B, but tree-backed doc A holds ranks 2-4:
    # A's aggregated vote (1/2+1/3+1/4 ~= 1.08) beats B's (1/1 = 1.0) -> pageindex wins.
    store = _store_with("A")
    candidates = [_cand("B"), _cand("A"), _cand("A"), _cand("A")]
    assert _vote_docs(candidates, store, Settings()) == ["A"]
    router = make_retrieval_router(store, Settings())
    assert router({"candidates": candidates}) == "pageindex"


def test_dominant_non_tree_source_keeps_hybrid_chunks():
    # Non-tree doc B wins the vote outright -> stay on hybrid chunks even though
    # tree-backed doc A appears in the top-k.
    store = _store_with("A")
    candidates = [_cand("B"), _cand("B"), _cand("B"), _cand("A"), _cand("A")]
    assert _vote_docs(candidates, store, Settings()) == []
    router = make_retrieval_router(store, Settings())
    assert router({"candidates": candidates}) == "evaluate"


def test_vote_window_is_bounded_by_k():
    # Doc A's chunks sit outside the vote window -> they cast no votes.
    store = _store_with("A")
    candidates = [_cand("B")] * 5 + [_cand("A")] * 10
    assert _vote_docs(candidates, store, Settings(pageindex_doc_vote_k=5)) == []

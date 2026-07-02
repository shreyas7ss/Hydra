"""Phase 2: the graph with a real retriever attached produces grounded candidates."""

from hydra.config import Settings
from hydra.graph import run_query
from hydra.llm import ScriptedLLM
from hydra.retrieval import HybridRetriever
from hydra.retrieval.documents import Document
from hydra.retrieval.embeddings import HashingEmbedder
from hydra.retrieval.rerank import LexicalReranker

DOCS = [
    Document(id="margin", text="operating margin for 2023 expanded to 21.5 percent from 18.2 in 2022"),
    Document(id="clause", text="clause 7.2 limitation of liability caps aggregate damages"),
    Document(id="pto", text="paid time off accrues at 1.5 days per month"),
]

_GEN_REFLECT = {
    "retrieval_eval": '{"confidence": "high", "score": 0.9, "reasoning": "on topic"}',
    "generate": "Operating margin rose to 21.5 percent in 2023.",
    "reflect": '{"faithful": true, "relevant": true, "critique": "grounded"}',
}

COMPLEX_LLM = ScriptedLLM({
    "intent_classification": '{"intent": "complex", "confidence": 0.8, "reasoning": "x"}',
    "multi_query": '["2023 operating margin", "margin expansion 2023"]',
    "decompose": '["what was operating margin in 2023?"]',
    "hyde": "operating margin in 2023 was 21.5 percent",
    **_GEN_REFLECT,
})

DIRECT_LLM = ScriptedLLM({
    "intent_classification": '{"intent": "direct", "confidence": 0.95, "reasoning": "exact id"}',
    **_GEN_REFLECT,
})


def _retriever():
    return HybridRetriever.from_documents(
        DOCS, embedder=HashingEmbedder(dim=128), reranker=LexicalReranker(), settings=Settings()
    )


def test_complex_path_returns_reranked_candidates():
    state = run_query("what was the operating margin in 2023?",
                      llm=COMPLEX_LLM, settings=Settings(), retriever=_retriever())
    assert state["retrieval_path"] == "hybrid_dense_bm25_rrf_rerank"
    assert state["candidates"]
    assert state["candidates"][0]["id"] == "margin"
    nodes = [s["node"] for s in state["trace"]]
    # Complex path runs the CRAG gate and generation to a grounded answer.
    assert nodes[:4] == ["route_intent", "transform_query", "hybrid_retrieve", "retrieval_evaluator"]
    assert "generate" in nodes
    assert state["answer"]
    assert state["citations"]
    assert state["reflection"]["ok"] is True


def test_direct_path_uses_bm25_fast_path():
    state = run_query("clause 7.2",
                      llm=DIRECT_LLM, settings=Settings(), retriever=_retriever())
    assert state["retrieval_path"] == "bm25_sql_direct"
    assert state["candidates"][0]["id"] == "clause"
    assert state["candidates"][0]["sources"] == ["bm25"]
    # Direct path still produces a grounded answer, skipping the CRAG evaluator.
    nodes = [s["node"] for s in state["trace"]]
    assert "retrieval_evaluator" not in nodes
    assert state["answer"]

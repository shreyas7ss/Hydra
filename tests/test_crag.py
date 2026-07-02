from hydra.config import Settings
from hydra.llm import ScriptedLLM
from hydra.nodes.crag import (
    ask_user,
    make_crag_router,
    make_retrieval_evaluator,
    make_secondary_retrieve,
)

CANDS = [{"id": "d1", "text": "operating margin 2023", "metadata": {"page": 1}}]


def test_evaluator_maps_llm_grade_to_confidence():
    llm = ScriptedLLM({"retrieval_eval": '{"confidence": "high", "score": 0.9, "reasoning": "x"}'})
    node = make_retrieval_evaluator(llm, Settings())
    out = node({"query": "margin?", "candidates": CANDS})
    assert out["retrieval_confidence"] == "high"
    assert out["retrieval_score"] == 0.9


def test_evaluator_short_circuits_on_empty_without_llm_call():
    llm = ScriptedLLM({"retrieval_eval": '{"confidence": "high", "score": 1.0}'})
    node = make_retrieval_evaluator(llm, Settings())
    out = node({"query": "margin?", "candidates": []})
    assert out["retrieval_confidence"] == "low"
    assert llm.calls == []  # no LLM spent on empty retrieval


def test_evaluator_defaults_medium_on_unparseable():
    llm = ScriptedLLM({"retrieval_eval": "not json"})
    node = make_retrieval_evaluator(llm, Settings())
    out = node({"query": "q", "candidates": CANDS})
    assert out["retrieval_confidence"] == "medium"


def test_crag_router_branches():
    router = make_crag_router(Settings(crag_max_retries=2))
    assert router({"retrieval_confidence": "high", "crag_retries": 0}) == "generate"
    assert router({"retrieval_confidence": "low", "crag_retries": 0}) == "secondary_retrieve"
    assert router({"retrieval_confidence": "medium", "crag_retries": 1}) == "secondary_retrieve"
    # retries exhausted
    assert router({"retrieval_confidence": "medium", "crag_retries": 2}) == "generate"
    assert router({"retrieval_confidence": "low", "crag_retries": 2}) == "ask_user"


def test_secondary_retrieve_increments_guard_without_retriever():
    node = make_secondary_retrieve(None, settings=Settings())
    out = node({"query": "q", "crag_retries": 0})
    assert out["crag_retries"] == 1


def test_ask_user_produces_clarification():
    out = ask_user({"query": "q"})
    assert out["answer"]
    assert out["citations"] == []

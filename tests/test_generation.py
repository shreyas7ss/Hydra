from hydra.config import Settings
from hydra.llm import ScriptedLLM
from hydra.nodes.generate import (
    make_generate,
    make_reflect_router,
    make_self_rag_reflect,
)

CANDS = [
    {"id": "margin", "text": "operating margin for 2023 was 21.5 percent",
     "metadata": {"source": "10-K", "page": 43, "section": "Margins"}},
]


def test_generate_produces_answer_and_citations():
    llm = ScriptedLLM({"generate": "Operating margin was 21.5 percent in 2023."})
    node = make_generate(llm, Settings())
    out = node({"query": "what was the margin?", "candidates": CANDS})
    assert out["answer"] == "Operating margin was 21.5 percent in 2023."
    assert out["citations"][0]["page"] == 43
    assert out["citations"][0]["section"] == "Margins"
    assert out["generation_count"] == 1


def test_generate_falls_back_without_context():
    node = make_generate(ScriptedLLM({"generate": "unused"}), Settings())
    out = node({"query": "q", "candidates": []})
    assert "don't have enough" in out["answer"].lower()
    assert out["citations"] == []


def test_reflect_marks_ok_and_not_ok():
    ok_llm = ScriptedLLM({"reflect": '{"faithful": true, "relevant": true, "critique": "good"}'})
    bad_llm = ScriptedLLM({"reflect": '{"faithful": false, "relevant": true, "critique": "hallucinated"}'})
    ok = make_self_rag_reflect(ok_llm, Settings())({"query": "q", "answer": "a", "candidates": CANDS})
    bad = make_self_rag_reflect(bad_llm, Settings())({"query": "q", "answer": "a", "candidates": CANDS})
    assert ok["reflection"]["ok"] is True
    assert bad["reflection"]["ok"] is False


def test_reflect_router_ends_or_regenerates_and_is_bounded():
    router = make_reflect_router(Settings(reflect_max_retries=1))
    # Faithful answer -> end.
    assert router({"reflection": {"ok": True}, "generation_count": 1}) == "end"
    # Unfaithful, budget remains -> regenerate.
    assert router({"reflection": {"ok": False}, "generation_count": 1}) == "regenerate"
    # Unfaithful, budget exhausted -> end (best effort).
    assert router({"reflection": {"ok": False}, "generation_count": 2}) == "end"

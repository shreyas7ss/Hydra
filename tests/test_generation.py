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


def test_pot_executes_program_and_finalizes():
    # First generate call emits a program; the finalize call (identified by the
    # "Computation result" marker in the user prompt) phrases the final answer.
    llm = ScriptedLLM({
        "Computation result": "Gross margin was 61.3%.",
        "generate": "```python\nrevenue = 1240  # p.42\ncogs = 480  # p.43\nanswer = round((revenue - cogs) / revenue * 100, 1)\n```",
    })
    node = make_generate(llm, Settings(enable_pot=True))
    out = node({"query": "what was gross margin?", "candidates": CANDS})
    assert out["answer"] == "Gross margin was 61.3%."
    assert "pot=ok (61.3)" in out["trace"][0]["detail"]


def test_pot_rejects_unsafe_program():
    llm = ScriptedLLM({
        "generate": "```python\nimport os\nanswer = 1\n```",
    })
    node = make_generate(llm, Settings(enable_pot=True))
    out = node({"query": "q", "candidates": CANDS})
    assert "pot=failed" in out["trace"][0]["detail"]
    assert "import" not in out["answer"]  # raw code never leaks into the answer


def test_pot_disabled_leaves_answer_untouched():
    llm = ScriptedLLM({"generate": "```python\nanswer = 2\n```"})
    node = make_generate(llm, Settings(enable_pot=False))
    out = node({"query": "q", "candidates": CANDS})
    assert "```python" in out["answer"]  # passthrough when the flag is off


def test_run_program_sandbox():
    from hydra.nodes.pot import run_program

    assert run_program("answer = round(21.5 - 18.2, 1)") == ("3.3", None)
    assert run_program("answer = __import__('os')")[1] is not None
    assert run_program("x = 1")[1] == "program did not set `answer`"
    assert run_program("answer = 1/0")[1].startswith("program error")


def test_reflect_router_ends_or_regenerates_and_is_bounded():
    router = make_reflect_router(Settings(reflect_max_retries=1))
    # Faithful answer -> end.
    assert router({"reflection": {"ok": True}, "generation_count": 1}) == "end"
    # Unfaithful, budget remains -> regenerate.
    assert router({"reflection": {"ok": False}, "generation_count": 1}) == "regenerate"
    # Unfaithful, budget exhausted -> end (best effort).
    assert router({"reflection": {"ok": False}, "generation_count": 2}) == "end"

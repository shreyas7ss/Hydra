from hydra.config import Settings
from hydra.graph import run_query
from hydra.llm import ScriptedLLM

COMPLEX_RESPONSES = {
    "intent_classification": '{"intent": "complex", "confidence": 0.8, "reasoning": "multi-hop"}',
    "multi_query": '["a", "b"]',
    "decompose": '["sub1", "sub2"]',
    "hyde": "hypothetical answer",
}

DIRECT_RESPONSES = {
    "intent_classification": '{"intent": "direct", "confidence": 0.95, "reasoning": "exact id"}',
    "generate": "The answer per the context.",
    "reflect": '{"faithful": true, "relevant": true, "critique": "ok"}',
}


def _trace_nodes(state):
    return [step["node"] for step in state["trace"]]


def test_complex_routes_through_transform_and_crag_gate():
    # No retriever attached -> the CRAG gate finds nothing and asks the user.
    state = run_query("compare 2022 and 2023 margins",
                      llm=ScriptedLLM(COMPLEX_RESPONSES), settings=Settings())
    assert state["intent"] == "complex"
    assert state["retrieval_path"] == "hybrid_dense_bm25_rrf_rerank"
    assert _trace_nodes(state)[:4] == [
        "route_intent", "transform_query", "hybrid_retrieve", "retrieval_evaluator",
    ]
    assert state["search_queries"][0] == "compare 2022 and 2023 margins"
    assert "ask_user" in _trace_nodes(state)
    assert state["answer"]  # clarification message


def test_direct_routes_to_fast_path_then_generates():
    state = run_query("clause 7.2",
                      llm=ScriptedLLM(DIRECT_RESPONSES), settings=Settings())
    assert state["intent"] == "direct"
    assert state["retrieval_path"] == "bm25_sql_direct"
    nodes = _trace_nodes(state)
    assert nodes[:3] == ["route_intent", "direct_lookup", "generate"]
    assert "transform_query" not in nodes  # fast path skips query transformation
    assert state["answer"]

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
}


def _trace_nodes(state):
    return [step["node"] for step in state["trace"]]


def test_complex_query_takes_transform_then_hybrid_path():
    state = run_query("compare 2022 and 2023 margins",
                      llm=ScriptedLLM(COMPLEX_RESPONSES), settings=Settings())
    assert state["intent"] == "complex"
    assert state["retrieval_path"] == "hybrid_dense_bm25_rrf_rerank"
    assert _trace_nodes(state) == ["route_intent", "transform_query", "hybrid_retrieve"]
    assert state["search_queries"][0] == "compare 2022 and 2023 margins"


def test_direct_query_takes_fast_path_and_skips_transforms():
    state = run_query("clause 7.2",
                      llm=ScriptedLLM(DIRECT_RESPONSES), settings=Settings())
    assert state["intent"] == "direct"
    assert state["retrieval_path"] == "bm25_sql_direct"
    assert _trace_nodes(state) == ["route_intent", "direct_lookup"]
    # transform_query never ran
    assert "expanded_queries" not in state or state["expanded_queries"] == []

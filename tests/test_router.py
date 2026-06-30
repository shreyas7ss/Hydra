from hydra.config import Settings
from hydra.llm import ScriptedLLM
from hydra.nodes.router import intent_router, make_route_intent


def _run(response: str, **overrides):
    settings = Settings(**overrides)
    llm = ScriptedLLM({"intent_classification": response})
    node = make_route_intent(llm, settings)
    return node({"query": "anything"})


def test_high_confidence_direct_stays_direct():
    out = _run('{"intent": "direct", "confidence": 0.9, "reasoning": "clause lookup"}')
    assert out["intent"] == "direct"
    assert out["intent_confidence"] == 0.9
    assert intent_router(out) == "direct"


def test_low_confidence_direct_is_escalated():
    out = _run('{"intent": "direct", "confidence": 0.2, "reasoning": "unsure"}',
               intent_confidence_floor=0.5)
    assert out["intent"] == "complex"
    assert "escalated" in out["intent_reasoning"]


def test_complex_is_preserved():
    out = _run('{"intent": "complex", "confidence": 0.8, "reasoning": "multi-hop"}')
    assert out["intent"] == "complex"


def test_unparseable_response_defaults_to_complex():
    out = _run("not json at all")
    assert out["intent"] == "complex"
    assert out["intent_confidence"] == 0.0


def test_response_wrapped_in_code_fence_is_parsed():
    out = _run('```json\n{"intent": "direct", "confidence": 0.95, "reasoning": "x"}\n```')
    assert out["intent"] == "direct"


def test_trace_entry_emitted():
    out = _run('{"intent": "complex", "confidence": 0.7, "reasoning": "x"}')
    assert out["trace"] == [{"node": "route_intent", "detail": "complex (conf=0.70)"}]

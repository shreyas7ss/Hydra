from hydra.config import Settings
from hydra.llm import ScriptedLLM
from hydra.nodes.transform import make_transform_query

RESPONSES = {
    "multi_query": '["revenue 2023", "annual sales 2023", "total income FY2023"]',
    "decompose": '["What was revenue in 2022?", "What was revenue in 2023?"]',
    "hyde": "The 2023 net revenue was reported as 1.2B in the annual filing.",
}


def test_all_transforms_assemble_search_fanout():
    settings = Settings(multi_query_count=3)
    node = make_transform_query(ScriptedLLM(RESPONSES), settings)
    out = node({"query": "net revenue 2023"})

    assert out["expanded_queries"] == ["revenue 2023", "annual sales 2023", "total income FY2023"]
    assert out["sub_queries"] == ["What was revenue in 2022?", "What was revenue in 2023?"]
    assert out["hyde_doc"].startswith("The 2023 net revenue")

    # Original query leads the fan-out, expansions + sub-queries follow, deduped.
    assert out["search_queries"][0] == "net revenue 2023"
    assert "annual sales 2023" in out["search_queries"]
    assert "What was revenue in 2022?" in out["search_queries"]
    # HyDE doc is an embedding seed, not added to the keyword fan-out.
    assert out["hyde_doc"] not in out["search_queries"]


def test_multi_query_count_is_capped():
    settings = Settings(multi_query_count=2)
    node = make_transform_query(ScriptedLLM(RESPONSES), settings)
    out = node({"query": "net revenue 2023"})
    assert len(out["expanded_queries"]) == 2


def test_flags_disable_transforms():
    settings = Settings(enable_multi_query=False, enable_decomposition=False, enable_hyde=False)
    node = make_transform_query(ScriptedLLM(RESPONSES), settings)
    out = node({"query": "net revenue 2023"})

    assert out["expanded_queries"] == []
    assert out["sub_queries"] == []
    assert out["hyde_doc"] == ""
    assert out["search_queries"] == ["net revenue 2023"]


def test_fanout_is_deduplicated():
    responses = {"multi_query": '["net revenue 2023", "NET REVENUE 2023"]',
                 "decompose": '["net revenue 2023"]', "hyde": "x"}
    settings = Settings(multi_query_count=3)
    node = make_transform_query(ScriptedLLM(responses), settings)
    out = node({"query": "net revenue 2023"})
    assert out["search_queries"] == ["net revenue 2023"]

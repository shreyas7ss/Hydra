from hydra.llm import ScriptedLLM
from hydra.pageindex.build import build_tree_from_sections, build_tree_from_text


def test_summaries_are_generated_per_node():
    llm = ScriptedLLM({"node_summary": "a concise summary"})
    tree = build_tree_from_sections(
        "Doc", [{"title": "S", "content": "some content", "level": 1}], llm
    )
    node = next(n for n in tree.iter_nodes() if n.title == "S")
    assert node.summary == "a concise summary"


def test_mode3_llm_segmentation_from_unstructured_text():
    llm = ScriptedLLM({
        "segment": '[{"title": "Alpha", "content": "aaa"}, {"title": "Beta", "content": "bbb"}]',
        "node_summary": "s",
    })
    tree = build_tree_from_text("Doc", "unstructured blob", llm)
    titles = [n.title for n in tree.iter_nodes()]
    assert "Alpha" in titles and "Beta" in titles


def test_mode3_falls_back_to_single_node_when_segmentation_fails():
    llm = ScriptedLLM({"segment": "not json", "node_summary": "s"})
    tree = build_tree_from_text("Doc", "raw text body", llm)
    # Always yields a usable tree: one flat node holding the text.
    content_nodes = [n for n in tree.iter_nodes() if n.content]
    assert content_nodes and "raw text body" in content_nodes[0].content


def test_nesting_respects_levels():
    sections = [
        {"title": "Top", "content": "t", "level": 1},
        {"title": "Sub", "content": "s", "level": 2},
    ]
    tree = build_tree_from_sections("Doc", sections, ScriptedLLM({"node_summary": "x"}))
    sub = next(n for n in tree.iter_nodes() if n.title == "Sub")
    assert sub.section_path == ["Top", "Sub"]

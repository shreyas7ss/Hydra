from hydra.llm import EchoLLM
from hydra.pageindex import build_tree_from_sections, flatten_to_documents, full_text
from hydra.sample_data import SAMPLE_SECTIONS


def _tree():
    return build_tree_from_sections("Doc", SAMPLE_SECTIONS, EchoLLM(), source="Doc")


def _node(tree, title):
    return next(n for n in tree.iter_nodes() if n.title == title)


def test_hierarchy_and_section_path():
    tree = _tree()
    margins = _node(tree, "Margins")
    assert margins.page == 43
    assert margins.section_path[-1] == "Margins"
    # nested under the level-2 "Results of Operations" section
    assert "Results of Operations" in margins.section_path


def test_full_text_keeps_table_and_footnote_together():
    tree = _tree()
    text = full_text(_node(tree, "Margins"))
    assert "21.5%" in text  # the table
    assert "Footnote 1" in text  # its footnote — never severed


def test_flatten_to_documents_carries_metadata():
    tree = _tree()
    docs = flatten_to_documents(tree)
    margins_doc = next(d for d in docs if d.metadata["section"].endswith("Margins"))
    assert margins_doc.metadata["source"] == "Doc"
    assert margins_doc.metadata["page"] == 43
    assert margins_doc.metadata["node_id"] == margins_doc.id

from hydra.config import Settings
from hydra.llm import EchoLLM
from hydra.pageindex import build_tree_from_sections, tree_search
from hydra.pageindex.search import TreeStore
from hydra.sample_data import SAMPLE_SECTIONS


def _tree():
    return build_tree_from_sections("Doc", SAMPLE_SECTIONS, EchoLLM(), source="Doc")


def test_traversal_reaches_margins_and_returns_intact_node():
    tree = _tree()
    results, path = tree_search(
        tree, "what was the operating margin and its footnote?", EchoLLM(), Settings()
    )
    assert results
    text = results[0].document.text
    # intact context: table + footnote together
    assert "21.5%" in text and "Footnote 1" in text
    # navigation descended to the Margins section
    assert path[-1] == "Margins"
    assert results[0].sources[0].startswith("pageindex:")


def test_traversal_metadata_points_to_tree_location():
    tree = _tree()
    results, _ = tree_search(tree, "operating margin footnote", EchoLLM(), Settings())
    meta = results[0].document.metadata
    assert meta["source"] == "Doc"
    assert meta["page"] == 43
    assert meta["section"].endswith("Margins")


def test_tree_store_lookup_by_source_and_doc_id():
    store = TreeStore()
    store.add(_tree())
    assert "Doc" in store
    assert store.get("Doc") is not None
    assert bool(store) is True

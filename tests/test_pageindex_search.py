from hydra.config import Settings
from hydra.llm import EchoLLM
from hydra.pageindex import build_tree_from_sections, tree_search
from hydra.pageindex.search import TreeStore
from hydra.sample_data import SAMPLE_SECTIONS


def _tree():
    return build_tree_from_sections("Doc", SAMPLE_SECTIONS, EchoLLM(), source="Doc")


def test_traversal_reaches_margins_and_returns_intact_node():
    tree = _tree()
    results, paths = tree_search(
        tree, "what was the operating margin and its footnote?", EchoLLM(), Settings()
    )
    assert results
    text = results[0].document.text
    # intact context: table + footnote together
    assert "21.5%" in text and "Footnote 1" in text
    # best navigation path descended to the Margins section
    assert paths[0].endswith("Margins")
    assert results[0].sources[0].startswith("pageindex:")


def test_beam_returns_multiple_candidate_nodes():
    tree = _tree()
    results, _ = tree_search(
        tree, "operating margin and revenue in 2023", EchoLLM(),
        Settings(pageindex_beam_width=2, pageindex_max_nodes=3),
    )
    # beam=2 hedges: both plausible sections come back, best first.
    ids = [r.document.id for r in results]
    assert len(ids) >= 2
    assert results[0].score >= results[1].score


def test_greedy_beam_width_one_returns_single_path():
    tree = _tree()
    results, paths = tree_search(
        tree, "operating margin footnote", EchoLLM(),
        Settings(pageindex_beam_width=1),
    )
    assert len(paths) >= 1
    assert paths[0].endswith("Margins")


def test_cross_reference_following():
    from hydra.pageindex.build import build_tree_from_sections

    sections = [
        {"title": "Results", "level": 1, "page": 10,
         "content": "Deferred revenue increased; see Note 7 for the breakdown."},
        {"title": "Note 7", "level": 1, "page": 55,
         "content": "Subscription balances totaled $310M at period end."},
    ]
    tree = build_tree_from_sections("Doc", sections, EchoLLM(), source="Doc")
    results, paths = tree_search(
        tree, "how did deferred revenue change in results?", EchoLLM(),
        Settings(pageindex_follow_refs=True, pageindex_beam_width=1),
    )
    texts = " ".join(r.document.text for r in results).lower()
    # the referenced Note 7 node is pulled in alongside the selected section
    assert "subscription balances" in texts
    assert any(p.startswith("xref") for p in paths)


def test_cross_reference_disabled_by_flag():
    from hydra.pageindex.build import build_tree_from_sections

    sections = [
        {"title": "Results", "level": 1, "page": 10,
         "content": "Deferred revenue increased; see Note 7 for the breakdown."},
        {"title": "Note 7", "level": 1, "page": 55,
         "content": "Note 7: subscription balances of $310M."},
    ]
    tree = build_tree_from_sections("Doc", sections, EchoLLM(), source="Doc")
    _, paths = tree_search(
        tree, "how did deferred revenue change in results?", EchoLLM(),
        Settings(pageindex_follow_refs=False, pageindex_beam_width=1, pageindex_max_nodes=1),
    )
    assert not any(p.startswith("xref") for p in paths)


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

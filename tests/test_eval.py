from hydra.config import Settings
from hydra.eval import evaluate, load_dataset, sample_golden_set
from hydra.eval.metrics import (
    answer_relevance_proxy,
    faithfulness_proxy,
    percentile,
    retrieval_metrics,
)
from hydra.llm import EchoLLM
from hydra.retrieval import HybridRetriever
from hydra.retrieval.embeddings import HashingEmbedder
from hydra.retrieval.rerank import LexicalReranker
from hydra.sample_data import sample_documents


# --- metrics ---
def test_retrieval_metrics_perfect_ranking():
    m = retrieval_metrics(["a", "b", "c"], ["a"], k_values=(1, 3))
    assert m["hit_rate@1"] == 1.0
    assert m["mrr"] == 1.0
    assert m["recall@3"] == 1.0
    assert m["precision@1"] == 1.0


def test_retrieval_metrics_relevant_at_rank_two():
    m = retrieval_metrics(["x", "a", "c"], ["a"], k_values=(1, 3))
    assert m["hit_rate@1"] == 0.0
    assert m["hit_rate@3"] == 1.0
    assert m["mrr"] == 0.5


def test_retrieval_metrics_miss():
    m = retrieval_metrics(["x", "y"], ["a"], k_values=(1, 3))
    assert m["hit_rate@3"] == 0.0
    assert m["mrr"] == 0.0


def test_percentile_interpolates():
    assert percentile([10, 20, 30, 40], 0.5) == 25.0
    assert percentile([5], 0.95) == 5


def test_generation_proxies():
    assert faithfulness_proxy("margin expanded", ["operating margin expanded in 2023"]) == 1.0
    assert faithfulness_proxy("totally unrelated words here", ["operating margin"]) < 0.5
    assert answer_relevance_proxy("the margin was 21.5 percent", "what was the margin?") > 0.0


# --- runner (end-to-end, offline) ---
def _retriever(settings):
    return HybridRetriever.from_documents(
        sample_documents(), embedder=HashingEmbedder(dim=256),
        reranker=LexicalReranker(), settings=settings,
    )


def test_evaluate_runs_over_golden_set():
    settings = Settings()
    report = evaluate(sample_golden_set(), llm=EchoLLM(), settings=settings,
                      retriever=_retriever(settings))
    assert len(report.per_query) == len(sample_golden_set())
    # The hybrid pipeline should find most gold docs in the top 3 on this easy set.
    assert report.aggregate["hit_rate@3"] >= 0.7
    # Ops accounting is populated.
    assert report.llm["calls"] > 0
    assert report.latency["p95"] >= report.latency["p50"]
    # Phase 4 generator is wired: generation metrics are now active.
    assert report.generation_active is True
    assert "faithfulness" in report.aggregate


def test_load_dataset_roundtrip(tmp_path):
    p = tmp_path / "gold.jsonl"
    p.write_text('{"query": "q", "relevant_ids": ["d1"]}\n', encoding="utf-8")
    ds = load_dataset(str(p))
    assert ds[0].query == "q"
    assert ds[0].relevant_ids == ["d1"]

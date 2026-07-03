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
    p.write_text(
        '{"query": "q", "relevant_ids": ["d1"], "answer": "42", "evidence_pages": [7]}\n',
        encoding="utf-8",
    )
    ds = load_dataset(str(p))
    assert ds[0].query == "q"
    assert ds[0].relevant_ids == ["d1"]
    assert ds[0].answer == "42"
    assert ds[0].evidence_pages == [7]


def test_evidence_page_recall_with_adjacency():
    from hydra.eval.metrics import evidence_page_recall

    assert evidence_page_recall([43, 10], [43]) == 1.0
    assert evidence_page_recall([44], [43]) == 1.0    # adjacent page counts
    assert evidence_page_recall([10], [43]) == 0.0
    assert evidence_page_recall([43, None], [43, 90]) == 0.5


def test_judge_correctness_numeric():
    from hydra.eval.runner import judge_correctness

    llm = EchoLLM()
    q = "what was the operating margin in 2023?"
    assert judge_correctness(llm, q, "21.5%", "The margin was 21.5 percent.") == 1.0
    assert judge_correctness(llm, q, "21.5%", "The margin was 18.2 percent.") == 0.0


def test_correctness_and_page_recall_flow_through_evaluate():
    from hydra.eval.dataset import EvalExample

    settings = Settings()
    ds = [EvalExample(
        query="what was the operating margin in 2023?",
        relevant_ids=["fin-2023-margin"],
        answer="21.5%",
        evidence_pages=[43],
    )]
    report = evaluate(ds, llm=EchoLLM(), settings=settings, retriever=_retriever(settings))
    assert "correctness" in report.aggregate
    assert "evidence_page_recall" in report.aggregate
    assert report.aggregate["evidence_page_recall"] == 1.0


def test_long_context_baseline_runs_and_judges():
    from hydra.eval.dataset import EvalExample
    from hydra.eval.runner import evaluate_long_context
    from hydra.sample_data import sample_documents

    ds = [EvalExample(query="what was the operating margin in 2023?", answer="21.5%")]
    corpus = "\n\n".join(d.text for d in sample_documents())
    report = evaluate_long_context(ds, llm=EchoLLM(), corpus_text=corpus)
    assert report.per_query[0].retrieval_path == "long-context"
    assert "correctness" in report.aggregate
    assert report.llm["calls"] >= 2  # generate + judge

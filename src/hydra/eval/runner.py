"""Evaluation runner.

Executes the graph over a dataset, collecting retrieval metrics, latency, and LLM
usage (the "Retrieval Tax": tokens + latency). Generation metrics are computed only
when a state carries an ``answer`` (i.e. once Phase 4 lands).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from time import perf_counter

from hydra.config import Settings
from hydra.eval.dataset import EvalExample
from hydra.eval.metrics import (
    answer_relevance_proxy,
    evidence_page_recall,
    faithfulness_proxy,
    percentile,
    retrieval_metrics,
)
from hydra.graph import run_query
from hydra.llm import LLMClient, parse_json

JUDGE_SYSTEM = """You are grading a question-answering system against a gold answer.
Judge whether the candidate answer is factually equivalent to the gold answer for the
question (numeric values must match after rounding; wording may differ; extra correct
detail is fine; missing or wrong values are incorrect).

Respond with ONLY JSON: {"correct": true|false, "reason": "<short>"}
TASK: judge"""


def judge_correctness(llm: LLMClient, question: str, gold: str, candidate: str) -> float:
    """LLM-as-judge correctness vs the gold answer. Returns 1.0 / 0.0."""
    user = f"Question: {question}\n\nGold answer: {gold}\n\nCandidate answer: {candidate}"
    data = parse_json(llm.complete(system=JUDGE_SYSTEM, user=user)) or {}
    return 1.0 if data.get("correct") is True else 0.0


class InstrumentedLLM:
    """Wraps any LLMClient to count calls and approximate token usage (chars/4)."""

    def __init__(self, inner: LLMClient) -> None:
        self.inner = inner
        self.calls = 0
        self.prompt_chars = 0
        self.completion_chars = 0

    def complete(self, *, system: str, user: str) -> str:
        self.calls += 1
        self.prompt_chars += len(system) + len(user)
        out = self.inner.complete(system=system, user=user)
        self.completion_chars += len(out)
        return out

    @property
    def approx_tokens(self) -> int:
        return (self.prompt_chars + self.completion_chars) // 4


@dataclass
class QueryResult:
    example: EvalExample
    ranked_ids: list[str]
    metrics: dict[str, float]
    latency_s: float
    intent: str | None
    retrieval_path: str | None


@dataclass
class EvalReport:
    per_query: list[QueryResult]
    aggregate: dict[str, float]
    latency: dict[str, float]
    llm: dict[str, float]
    k_values: tuple[int, ...]
    generation_active: bool = False
    per_query_extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_queries": len(self.per_query),
            "aggregate": self.aggregate,
            "latency_s": self.latency,
            "llm": self.llm,
            "generation_active": self.generation_active,
            "per_query": [
                {
                    "query": r.example.query,
                    "intent": r.intent,
                    "retrieval_path": r.retrieval_path,
                    "ranked_ids": r.ranked_ids,
                    "relevant_ids": r.example.relevant_ids,
                    "metrics": r.metrics,
                    "latency_s": round(r.latency_s, 4),
                }
                for r in self.per_query
            ],
        }


def evaluate(
    dataset: list[EvalExample],
    *,
    llm: LLMClient,
    settings: Settings,
    retriever=None,
    tree_store=None,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
) -> EvalReport:
    instrumented = InstrumentedLLM(llm)
    results: list[QueryResult] = []
    generation_active = False

    for ex in dataset:
        t0 = perf_counter()
        state = run_query(ex.query, llm=instrumented, settings=settings,
                          retriever=retriever, tree_store=tree_store)
        latency = perf_counter() - t0

        candidates = state.get("candidates", []) or []
        ranked_ids = [c["id"] for c in candidates]
        metrics = retrieval_metrics(ranked_ids, ex.relevant_ids, k_values)

        if ex.evidence_pages:
            pages = [(c.get("metadata") or {}).get("page") for c in candidates]
            metrics["evidence_page_recall"] = evidence_page_recall(pages, ex.evidence_pages)

        answer = state.get("answer")
        if answer:
            generation_active = True
            contexts = [c["text"] for c in candidates]
            metrics["faithfulness"] = faithfulness_proxy(answer, contexts)
            metrics["answer_relevance"] = answer_relevance_proxy(answer, ex.query)
            if ex.answer:
                metrics["correctness"] = judge_correctness(instrumented, ex.query, ex.answer, answer)

        results.append(
            QueryResult(
                example=ex,
                ranked_ids=ranked_ids,
                metrics=metrics,
                latency_s=latency,
                intent=state.get("intent"),
                retrieval_path=state.get("retrieval_path"),
            )
        )

    return _build_report(results, instrumented, k_values, generation_active)


def _build_report(
    results: list[QueryResult],
    instrumented: InstrumentedLLM,
    k_values: tuple[int, ...],
    generation_active: bool,
) -> EvalReport:
    # Aggregate each metric key across queries (only where present).
    metric_keys: list[str] = []
    for r in results:
        for key in r.metrics:
            if key not in metric_keys:
                metric_keys.append(key)
    aggregate = {
        key: mean([r.metrics[key] for r in results if key in r.metrics])
        for key in metric_keys
    }

    latencies = [r.latency_s for r in results]
    latency = {
        "mean": mean(latencies) if latencies else 0.0,
        "p50": percentile(latencies, 0.50),
        "p95": percentile(latencies, 0.95),
        "max": max(latencies) if latencies else 0.0,
    }

    n = max(1, len(results))
    llm_stats = {
        "calls": float(instrumented.calls),
        "approx_tokens": float(instrumented.approx_tokens),
        "calls_per_query": instrumented.calls / n,
        "approx_tokens_per_query": instrumented.approx_tokens / n,
    }

    return EvalReport(
        per_query=results,
        aggregate=aggregate,
        latency=latency,
        llm=llm_stats,
        k_values=k_values,
        generation_active=generation_active,
    )


LONG_CONTEXT_SYSTEM = """You are answering questions about the document(s) provided in
full. Answer concisely using only the document contents; compute carefully when the
question requires arithmetic.

TASK: generate"""


def evaluate_long_context(
    dataset: list[EvalExample],
    *,
    llm: LLMClient,
    corpus_text: str,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
) -> EvalReport:
    """Baseline: no retrieval — the entire corpus is stuffed into every prompt.

    The strong-but-expensive comparator any retrieval claim must beat on accuracy
    or crush on tokens/latency. Retrieval metrics are meaningless here and omitted.
    """
    instrumented = InstrumentedLLM(llm)
    results: list[QueryResult] = []

    for ex in dataset:
        t0 = perf_counter()
        answer = instrumented.complete(
            system=LONG_CONTEXT_SYSTEM,
            user=f"Question: {ex.query}\n\nContext:\n{corpus_text}",
        ).strip()
        latency = perf_counter() - t0

        metrics: dict[str, float] = {}
        if ex.answer:
            metrics["correctness"] = judge_correctness(instrumented, ex.query, ex.answer, answer)
        results.append(
            QueryResult(
                example=ex, ranked_ids=[], metrics=metrics,
                latency_s=latency, intent="long-context", retrieval_path="long-context",
            )
        )

    return _build_report(results, instrumented, k_values, generation_active=True)

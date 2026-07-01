"""Evaluation metrics.

Retrieval metrics are computed from the ranked candidate IDs vs the gold set. Generation
proxies are deterministic lexical stand-ins for LLM-judged faithfulness/relevance — good
enough for regression tracking offline; swap in an LLM judge (or RAGAS) for headline numbers.
"""

from __future__ import annotations

import math

from hydra.retrieval.text import tokenize


def retrieval_metrics(
    ranked_ids: list[str],
    relevant_ids: list[str],
    k_values: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, float]:
    relevant = set(relevant_ids)
    metrics: dict[str, float] = {}

    for k in k_values:
        topk = ranked_ids[:k]
        hit_set = set(topk) & relevant
        metrics[f"hit_rate@{k}"] = 1.0 if hit_set else 0.0
        metrics[f"recall@{k}"] = (len(hit_set) / len(relevant)) if relevant else 0.0
        metrics[f"precision@{k}"] = (len(hit_set) / k) if k else 0.0

    # Mean Reciprocal Rank over the full ranking (rank of first relevant hit).
    rr = 0.0
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            rr = 1.0 / rank
            break
    metrics["mrr"] = rr
    return metrics


def faithfulness_proxy(answer: str, contexts: list[str]) -> float:
    """Fraction of answer terms supported by the retrieved contexts (grounding)."""
    a_terms = set(tokenize(answer))
    if not a_terms:
        return 0.0
    ctx_terms = set(tokenize(" ".join(contexts)))
    return len(a_terms & ctx_terms) / len(a_terms)


def answer_relevance_proxy(answer: str, question: str) -> float:
    """Fraction of question terms addressed by the answer."""
    q_terms = set(tokenize(question))
    if not q_terms:
        return 0.0
    a_terms = set(tokenize(answer))
    return len(q_terms & a_terms) / len(q_terms)


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile; p in [0, 1]."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)

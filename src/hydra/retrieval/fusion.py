"""Reciprocal Rank Fusion (RRF), k=60.

Combines an arbitrary number of ranked result lists (dense per query, BM25 per query,
dense-over-HyDE) into one ranking using only rank position — robust to the wildly
different score scales of cosine vs BM25. k=60 is the directive-mandated constant.
"""

from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(
    labeled_lists: list[tuple[str, list[tuple[str, float]]]],
    *,
    k: int = 60,
) -> tuple[list[tuple[str, float]], dict[str, list[str]]]:
    """Fuse ``(label, ranked [(doc_id, score)])`` lists.

    Returns ``(fused, contributors)`` where ``fused`` is ``[(doc_id, rrf_score)]`` sorted
    best-first, and ``contributors`` maps each doc_id to the labels that surfaced it
    (kept for the audit trail).
    """
    scores: dict[str, float] = defaultdict(float)
    contributors: dict[str, set[str]] = defaultdict(set)

    for label, ranked in labeled_lists:
        for rank, (doc_id, _score) in enumerate(ranked):
            scores[doc_id] += 1.0 / (k + rank + 1)
            contributors[doc_id].add(label)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    contrib_sorted = {doc_id: sorted(labels) for doc_id, labels in contributors.items()}
    return fused, contrib_sorted

"""BM25 sparse index (Okapi BM25), pure-Python.

The directive mandates BM25 for exact identifiers (clause numbers, SKUs) where dense
semantic approximation fails. Implemented in-process to keep it transparent and
dependency-free; for very large corpora this would be backed by a real inverted index
(e.g. Elasticsearch/OpenSearch).
"""

from __future__ import annotations

import math
from collections import Counter

from hydra.retrieval.documents import Document
from hydra.retrieval.text import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._doc_ids: list[str] = []
        self._tokens: list[list[str]] = []
        self._tf: list[Counter[str]] = []
        self._doc_len: list[int] = []
        self._df: Counter[str] = Counter()
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0

    def build(self, documents: list[Document]) -> None:
        self._doc_ids = [d.id for d in documents]
        self._tokens = [tokenize(d.text) for d in documents]
        self._tf = [Counter(toks) for toks in self._tokens]
        self._doc_len = [len(toks) for toks in self._tokens]
        n = len(documents)
        self._avgdl = (sum(self._doc_len) / n) if n else 0.0

        self._df = Counter()
        for toks in self._tokens:
            for term in set(toks):
                self._df[term] += 1

        # BM25 idf with the standard +0.5 smoothing.
        self._idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for term, df in self._df.items()
        }

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if not self._doc_ids:
            return []
        q_terms = tokenize(query)
        scores: list[tuple[str, float]] = []
        for idx, doc_id in enumerate(self._doc_ids):
            tf = self._tf[idx]
            dl = self._doc_len[idx]
            denom_norm = self.k1 * (1 - self.b + self.b * (dl / self._avgdl)) if self._avgdl else self.k1
            score = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                idf = self._idf.get(term, 0.0)
                score += idf * (f * (self.k1 + 1)) / (f + denom_norm)
            if score > 0:
                scores.append((doc_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

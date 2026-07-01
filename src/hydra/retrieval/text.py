"""Shared tokenisation.

One tokenizer feeds BM25, the hashing embedder, and the lexical reranker so their
notions of a "term" stay consistent.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A deliberately small stoplist — enough to de-noise BM25/overlap scoring without
# dropping domain identifiers.
_STOPWORDS = frozenset(
    """
    a an the of to in on at for and or but is are was were be been being
    this that these those it its as by with from into than then so such
    what which who whom whose when where why how do does did
    """.split()
)


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    if drop_stopwords:
        return [t for t in tokens if t not in _STOPWORDS]
    return tokens

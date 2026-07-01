"""Dense embedders behind one protocol.

* ``OpenAIEmbedder``  — the directive baseline (text-embedding-3-*), lazy import.
* ``HashingEmbedder`` — deterministic, offline, no model download. A hashed bag-of-words
  vector: not semantically deep, but it makes shared-vocabulary docs measurably similar,
  which is enough to exercise the dense path and keep tests/demo runnable with no network.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol, runtime_checkable

from hydra.retrieval.text import tokenize


@runtime_checkable
class Embedder(Protocol):
    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Hash each token into one of ``dim`` buckets, then L2-normalise."""

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _bucket(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self._dim
            for token in tokenize(text):
                vec[self._bucket(token)] += 1.0
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            vectors.append(vec)
        return vectors


class OpenAIEmbedder:
    def __init__(self, model: str, api_key: str | None, dim: int | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError(
                "The 'openai' extra is not installed. Run `uv sync --extra openai`, "
                "or use the offline hashing embedder (HYDRA_EMBEDDING_PROVIDER=hashing)."
            ) from exc
        self._client = OpenAI(api_key=api_key)
        self.model = model
        # text-embedding-3-* support the `dimensions` parameter; default to model max.
        self._dim = dim or 3072

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        kwargs = {"model": self.model, "input": texts}
        if self._dim:
            kwargs["dimensions"] = self._dim
        resp = self._client.embeddings.create(**kwargs)
        return [item.embedding for item in resp.data]


def build_embedder(settings, *, demo: bool = False) -> Embedder:
    provider = "hashing" if demo else settings.embedding_provider.lower()
    if provider == "hashing":
        return HashingEmbedder(dim=settings.embedding_dim)
    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set for the OpenAI embedder. Set it in .env, "
                "or use HYDRA_EMBEDDING_PROVIDER=hashing (offline)."
            )
        # embedding_dim configures the offline hashing embedder; let OpenAI use its
        # model-default dimensionality.
        return OpenAIEmbedder(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )
    raise RuntimeError(f"Unknown embedding provider: {settings.embedding_provider!r}")

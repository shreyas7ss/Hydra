"""Runtime configuration, sourced from environment / .env.

A plain dataclass (no pydantic-settings dependency) keeps this readable and the
import graph light. Feature flags let us A/B each query transform against the eval
harness — the plan mandates keeping only the transforms that measurably win.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # LLM provider — swappable (plan §8 decision #2). The graph is provider-agnostic.
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"
    openai_api_key: str | None = None
    temperature: float = 0.0

    # Phase 1 query-transform feature flags
    enable_multi_query: bool = True
    enable_decomposition: bool = True
    enable_hyde: bool = True
    multi_query_count: int = 3

    # Routing: a "direct" classification below this confidence is escalated to the
    # thorough (complex) path. We would rather over-serve than ground an answer on
    # an under-retrieved fast-path result.
    intent_confidence_floor: float = 0.5

    # --- Phase 2: Coarse Retrieval & Precision Fusion ---
    embedding_provider: str = "openai"            # swappable; demo -> "hashing"
    embedding_model: str = "text-embedding-3-large"
    embedding_dim: int = 256                       # used by the offline hashing embedder
    reranker_provider: str = "lexical"             # "cross-encoder" needs the `rerank` extra
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    enable_dense: bool = True
    enable_sparse: bool = True
    retrieval_top_k: int = 10                      # final candidates returned to the generator
    per_query_top_k: int = 20                      # candidates pulled per retriever per query
    fusion_k: int = 60                             # Reciprocal Rank Fusion constant (k=60)
    rerank_pool: int = 50                          # top-N fused docs sent to the reranker

    # --- Phase 4: Corrective RAG + Self-RAG ---
    crag_max_retries: int = 2                      # secondary-retrieval attempts before giving up
    reflect_max_retries: int = 1                   # regenerations allowed on unfaithful answers
    generation_context_k: int = 5                  # top candidates passed to the generator

    @classmethod
    def from_env(cls) -> "Settings":
        # Load .env if python-dotenv is available; never hard-fail without it.
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except Exception:
            pass

        return cls(
            llm_provider=os.getenv("HYDRA_LLM_PROVIDER", "openai"),
            llm_model=os.getenv("HYDRA_LLM_MODEL", "gpt-4o"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=float(os.getenv("HYDRA_TEMPERATURE", "0.0")),
            enable_multi_query=_flag("HYDRA_ENABLE_MULTI_QUERY", True),
            enable_decomposition=_flag("HYDRA_ENABLE_DECOMPOSITION", True),
            enable_hyde=_flag("HYDRA_ENABLE_HYDE", True),
            multi_query_count=int(os.getenv("HYDRA_MULTI_QUERY_COUNT", "3")),
            intent_confidence_floor=float(os.getenv("HYDRA_INTENT_CONFIDENCE_FLOOR", "0.5")),
            embedding_provider=os.getenv("HYDRA_EMBEDDING_PROVIDER", "openai"),
            embedding_model=os.getenv("HYDRA_EMBEDDING_MODEL", "text-embedding-3-large"),
            embedding_dim=int(os.getenv("HYDRA_EMBEDDING_DIM", "256")),
            reranker_provider=os.getenv("HYDRA_RERANKER_PROVIDER", "lexical"),
            reranker_model=os.getenv("HYDRA_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            enable_dense=_flag("HYDRA_ENABLE_DENSE", True),
            enable_sparse=_flag("HYDRA_ENABLE_SPARSE", True),
            retrieval_top_k=int(os.getenv("HYDRA_RETRIEVAL_TOP_K", "10")),
            per_query_top_k=int(os.getenv("HYDRA_PER_QUERY_TOP_K", "20")),
            fusion_k=int(os.getenv("HYDRA_FUSION_K", "60")),
            rerank_pool=int(os.getenv("HYDRA_RERANK_POOL", "50")),
            crag_max_retries=int(os.getenv("HYDRA_CRAG_MAX_RETRIES", "2")),
            reflect_max_retries=int(os.getenv("HYDRA_REFLECT_MAX_RETRIES", "1")),
            generation_context_k=int(os.getenv("HYDRA_GENERATION_CONTEXT_K", "5")),
        )

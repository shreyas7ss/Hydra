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
        )

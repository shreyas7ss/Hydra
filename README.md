# Hydra — Hybrid Adaptive RAG

A vector + vectorless Retrieval-Augmented Generation system orchestrated with
**LangGraph**. See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the full roadmap.

## Status

**Phase 1 — Query Intelligence Layer** is implemented:

- **Intent Classification Gate** (`route_intent`) routes each query to a fast
  direct-lookup path or the agentic reasoning path, escalating low-confidence
  "direct" calls to the thorough path.
- **Query transformation** (`transform_query`): Multi-Query Expansion, Sub-Query
  Decomposition, and HyDE — each independently flag-gated for A/B evaluation.

Downstream retrieval (`hybrid_retrieve`, `direct_lookup`) are stubs until Phase 2.

## Quickstart

```bash
uv sync                       # install core deps (langgraph, dotenv)
uv sync --extra openai        # add the OpenAI provider (optional)

# Offline demo — no API key required:
uv run hydra --demo "compare 2022 and 2023 operating margins"
uv run hydra --demo "clause 7.2"

# With a configured provider (copy .env.example -> .env, set OPENAI_API_KEY):
uv run hydra "what was net revenue in 2023?"
```

## Develop / test

```bash
uv run pytest
```

## Layout

```
src/hydra/
  state.py            RAGState — the shared graph contract
  config.py           Settings (env-driven, feature flags)
  llm.py              LLMClient protocol; OpenAI + offline demo + scripted-test impls
  graph.py            LangGraph assembly (build_frontend_graph / run_query)
  nodes/
    router.py         route_intent + intent_router (the gate)
    transform.py      transform_query (multi-query / decompose / HyDE)
    downstream.py     hybrid_retrieve / direct_lookup stubs (Phase 2)
  cli.py              `hydra` entry point
tests/                router / transform / graph tests (offline, deterministic)
```

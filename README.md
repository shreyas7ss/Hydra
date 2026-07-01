# Hydra — Hybrid Adaptive RAG

A vector + vectorless Retrieval-Augmented Generation system orchestrated with
**LangGraph**. See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the full roadmap.

## Status

**Phase 1 — Query Intelligence Layer** (done):

- **Intent Classification Gate** (`route_intent`) routes each query to a fast
  direct-lookup path or the agentic reasoning path, escalating low-confidence
  "direct" calls to the thorough path.
- **Query transformation** (`transform_query`): Multi-Query Expansion, Sub-Query
  Decomposition, and HyDE — each independently flag-gated for A/B evaluation.

**Phase 2 — Coarse Retrieval & Precision Fusion** (done):

- **Hybrid retrieval** (`hybrid_retrieve`): dense (Qdrant HNSW) + BM25 sparse, one
  search per query in the Phase 1 fan-out, plus a HyDE-seeded dense search.
- **Fusion + rerank**: Reciprocal Rank Fusion (k=60) → rerank the top pool against
  the original query. Cross-encoder in production (via the `rerank` extra); a
  lightweight lexical reranker offline.
- **Direct fast path** (`direct_lookup`): BM25-only lookup for exact-identifier queries.
- Every candidate carries page/section metadata + which retriever lists surfaced it
  (audit trail).

Providers/models are swappable behind protocols; the whole stack runs offline (hashing
embedder + lexical reranker + demo LLM) so nothing requires an API key to develop/test.

## Quickstart

```bash
uv sync                       # core deps (langgraph, qdrant-client, dotenv)
uv sync --extra openai        # add OpenAI LLM + embeddings (optional)
uv sync --extra rerank        # add the cross-encoder reranker (heavy: torch)

# Offline demo — no API key required (real retrieval over a bundled sample corpus):
uv run hydra --demo "how did operating margin change from 2022 to 2023 and why?"
uv run hydra --demo "clause 7.2 limitation of liability"

# Point at your own corpus (JSONL: {"id","text","metadata"} per line):
uv run hydra --demo --corpus docs.jsonl "your question"

# With configured providers (copy .env.example -> .env, set OPENAI_API_KEY):
uv run hydra --corpus docs.jsonl "what was net revenue in 2023?"
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
    downstream.py     hybrid_retrieve / direct_lookup (retriever-backed, stub if none)
  retrieval/          Phase 2
    text.py           shared tokenizer
    documents.py      Document / ScoredDoc
    embeddings.py     Embedder protocol; OpenAI + offline Hashing
    dense.py          Qdrant HNSW dense index
    sparse.py         BM25 (Okapi) index
    fusion.py         Reciprocal Rank Fusion (k=60)
    rerank.py         Reranker protocol; CrossEncoder + offline Lexical
    pipeline.py       HybridRetriever (index -> multi-query search -> RRF -> rerank)
  sample_data.py      bundled demo corpus
  cli.py              `hydra` entry point
tests/                router / transform / graph / retrieval tests (offline, deterministic)
```

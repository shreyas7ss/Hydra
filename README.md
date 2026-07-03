# Hydra — Hybrid Adaptive RAG

A vector + vectorless Retrieval-Augmented Generation system orchestrated with


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

**Phase 3 — Vectorless / PageIndex** (done):

- **Hybrid as coarse filter, then tree.** `hybrid_retrieve` narrows to the candidate
  document(s); if a PageTree exists, `pageindex_tree_search` navigates Root→Section→Page
  and returns **intact nodes** (a section with its table + footnote together), never fragments.
- **Real PDF ingestion** (`--pdf`) via `pdfplumber`, with a **TOC fallback cascade**
  (page-numbered → un-numbered → pure-LLM segmentation) and per-node LLM summaries.

**Phase 4 — Corrective RAG (CRAG) + generation + Self-RAG** (done):

- **CRAG quality gate** (`retrieval_evaluator`): an LLM grades retrieval high/medium/low.
  High generates directly; medium/low triggers a bounded `secondary_retrieve` loop; exhausted
  low confidence routes to `ask_user` rather than answering on weak evidence.
- **Grounded generation** (`generate`): answers strictly from retrieved context and emits
  citations (source/page/section).
- **Self-RAG reflection** (`self_rag_reflect`): an LLM self-critique (faithful? relevant?);
  an unfaithful answer loops back for a bounded regeneration.

**Benchmark hardening** (for FinanceBench-class evaluation):

- **Program-of-thought generation**: arithmetic questions produce a sandboxed, executed
  Python program instead of mental math (`HYDRA_ENABLE_POT`).
- **Layout-aware PDF parsing**: font-size heading detection + running-header/footer removal.
- **Beam tree search** (`HYDRA_PAGEINDEX_BEAM_WIDTH`), multi-node returns, and
  "see Note X" **cross-reference following** (`HYDRA_PAGEINDEX_FOLLOW_REFS`).
- **Doc-vote routing**: the target document wins by aggregated top-k chunk votes, not a
  single lucky chunk.
- **Eval**: evidence-page recall, correctness-vs-gold (LLM judge), and a no-retrieval
  **long-context baseline** (`hydra-eval --baseline long-context`).

Providers/models are swappable behind protocols; the whole stack runs offline (hashing
embedder + lexical reranker + demo LLM) so nothing requires an API key to develop/test.

## Quickstart

```bash
uv sync                       # core deps (langgraph, qdrant-client, pdfplumber, dotenv)
uv sync --extra openai        # add OpenAI LLM + embeddings (optional)
uv sync --extra gemini        # add Google Gemini LLM + embeddings (optional)
uv sync --extra rerank        # add the cross-encoder reranker (heavy: torch)

# Offline demo — no API key required (real retrieval over a bundled sample corpus + tree):
uv run hydra --demo "what drove the change in operating margin, including the footnote?"
uv run hydra --demo "clause 7.2 limitation of liability"

# Ingest a real PDF (parse -> PageIndex tree -> navigate):
uv run hydra --demo --pdf report.pdf "what quarterly dividend was declared?"

# With configured providers (copy .env.example -> .env; set GOOGLE_API_KEY or OPENAI_API_KEY):
uv run hydra --corpus docs.jsonl "what was net revenue in 2023?"
```

**LLM provider** is set by `HYDRA_LLM_PROVIDER` (`openai` or `gemini`); the model default is
provider-aware (`gpt-4o` / `gemini-2.5-flash`). Set `HYDRA_EMBEDDING_PROVIDER=gemini` to run
retrieval on Gemini embeddings too. Everything also runs fully offline via `--demo`.

## Evaluate (Phase 0 harness)

Score retrieval quality + the "Retrieval Tax" (latency, tokens) against a golden set:

```bash
uv run hydra-eval --demo                       # built-in golden set over the sample corpus
uv run hydra-eval --demo --dataset gold.jsonl --corpus docs.jsonl
uv run hydra-eval --demo --min-hit-rate 0.7    # CI regression gate (non-zero exit on fail)
```

Reports Hit Rate / MRR / recall@k / precision@k, latency p50/p95, and LLM calls/tokens.
Generation metrics (faithfulness, answer-relevance) activate once Phase 4 produces answers.
Flip `HYDRA_ENABLE_DENSE` / `HYDRA_ENABLE_SPARSE` to A/B retrieval configurations.

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
    crag.py           retrieval_evaluator / secondary_retrieve / ask_user / crag_router
    generate.py       generate / self_rag_reflect / reflect_router
    pageindex.py      pageindex_tree_search / route_retrieval
  pageindex/          Phase 3
    tree.py           PageNode / PageTree / full_text / flatten_to_documents
    parse.py          pdfplumber PDF parsing -> sections
    build.py          tree construction + TOC fallback cascade + LLM summaries
    search.py         TreeStore + reasoning-based tree traversal
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

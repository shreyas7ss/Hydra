# Hydra — Hybrid Adaptive RAG: Implementation Plan

Derived from *Project Plan: Large-Scale Hybrid Adaptive RAG Implementation*. This document
re-frames the directive as a buildable, measurable engineering roadmap. Where the directive
states marketing-grade claims (e.g. "98.7% accuracy", "73% of failures"), this plan treats
them as **targets to be validated**, not assumptions.

**Orchestration framework: LangGraph.** The whole system is a *stateful, branching, cyclic*
workflow — adaptive routing, CRAG retry loops, and Self-RAG reflection are all decisions and
cycles, not linear chains. LangGraph models this natively as a graph of nodes over a shared
state object, with conditional edges and loops. We use LangChain only as a library for component
adapters (retrievers, embeddings, vector stores) where convenient — not as the control flow.

---

## 0. Guiding principles & honest caveats

Before any code, three realities shape everything below:

1. **Measure first, optimize second.** Every "precision gain" the directive cites (cross-encoder
   recall 0.69→0.81, 98.7% FinanceBench) is only meaningful against *our* corpus and *our* eval
   set. Phase 0 builds the eval harness so every later phase is judged, not assumed.

2. **Build the cheap baseline before the expensive stack.** The full directive (HNSW + BM25 + RRF +
   cross-encoder + PageIndex tree search + CRAG + Self-RAG + LLMLingua + Proxy-Pointer) is a lot of
   moving parts. Most of the accuracy usually comes from **hybrid search + a reranker**. We ship that
   first, measure, then add each adaptive layer only if the eval shows it pays for its latency/cost.

3. **Data sovereignty is a gate, not a footnote.** The directive's own CRITICAL warning says PageIndex
   ships *entire documents* to OpenAI during indexing *and* retrieval. That contradicts the HIPAA/SOC2
   objective and the "cloud is baseline" line. **Decision required before Phase 3** (see §8).

---

## 1. Architecture as a LangGraph state machine

A single graph over a shared `RAGState` (query, rewrites, candidates, scores, draft, citations,
retry counters). Nodes are functions; edges are conditional/cyclic.

```
                       ┌──────────────────────────────┐
   query ─► [route_intent] ──direct──► [bm25_sql_lookup] ─┐
                       │                                   │
                       └──complex──► [transform_query]     │
                                         │                 │
                                         ▼                 │
                              [hybrid_retrieve]            │
                              (dense HNSW + BM25           │
                               → RRF k=60 → cross-encoder) │
                                         │                 │
                          structured? ───┼───► [pageindex_tree_search]
                                         ▼                 │
                                 [retrieval_evaluator] ◄───┘
                                   │     │      │
                              high │ med │  low │
                                   ▼     ▼      ▼
                              [generate]  [secondary_retrieve]  [restart / ask_user]
                                   │            │  (loops back to evaluator)
                                   ▼
                          [self_rag_reflect] ──not faithful?──► loops back to [generate]
                                   │ ok
                                   ▼
                          answer + citations (page/section/node IDs)
```

LangGraph specifics we'll lean on:
- **Conditional edges** for the intent router and the CRAG high/med/low branch.
- **Cycles** for CRAG secondary retrieval and Self-RAG regeneration (with a max-iteration guard in state
  to prevent infinite loops).
- **Checkpointer** for per-step state persistence → gives us the audit trail (page/section/node IDs at
  every hop) the regulated-environment objective requires.
- **Streaming + per-node tracing** (LangSmith optional) for latency/token attribution per node.

---

## 2. Phase 0 — Foundation & Evaluation Harness *(do this first)*

**Goal:** repo skeleton, the LangGraph state contract, and a way to score quality before optimizing.

- Repo scaffold: Python package, dependency management (`uv`/`poetry`), config layer, secrets handling.
- Define `RAGState` (TypedDict/Pydantic) and a trivial 2-node graph end-to-end to prove the harness.
- Pick a representative **document set** (directive targets financial/legal/structured docs).
- Build a **golden eval set**: ~50–200 Q&A pairs with known source spans (page/section).
- Implement metrics from the directive's success list:
  - Retrieval: **Hit Rate**, **MRR**, recall@k.
  - Generation: **Faithfulness** (grounding), **Answer relevance**, **Noise robustness**.
  - Ops: **latency** (p50/p95) and **token cost** per query — to quantify the "Retrieval Tax", measured per LangGraph node.
- Wire an offline eval runner (e.g. RAGAS or a small custom harness) into CI.

**Exit criteria:** one-command `eval` printing retrieval + generation + per-node cost/latency.

---

## 3. Phase 1 — Adaptive Front-End (Query Intelligence Layer)

Implemented as the entry nodes + first conditional edge of the graph.

- **`route_intent` node + conditional edge**: `direct lookup` → `bm25_sql_lookup` fast path;
  `complex/multi-hop` → `transform_query`. Start with a cheap LLM/classifier; measure routing
  accuracy against labeled queries.
- **`transform_query` node** (add transforms incrementally, A/B each against eval):
  - Multi-Query Expansion (parallel variant searches).
  - Sub-Query Decomposition (multi-hop) — natural fit for a LangGraph sub-graph / fan-out.
  - HyDE (hypothetical-answer embedding seed).

**Exit criteria:** router sends ≥X% of simple queries down the cheap path with no accuracy loss;
each transform shows a measurable Hit Rate/MRR delta (keep only the winners).

---

## 4. Phase 2 — Coarse Retrieval & Precision Fusion *(highest ROI — ship early)*

The `hybrid_retrieve` node.

- Vector store with **HNSW** index (Qdrant or Pinecone — pick one in §8).
- **BM25 sparse** index for exact identifiers (SKUs, clause numbers) — directive notes BM25 often
  beats dense on financial/technical terms.
- **Reciprocal Rank Fusion at k=60** over dense + sparse results.
- **Cross-Encoder reranker** on top-50 fused candidates (directive's single largest precision gain).

**Exit criteria:** hybrid+rerank beats a dense-only baseline on our eval (validate the 0.69→0.81-style
lift on *our* data). This node alone should hit most of the accuracy target for many query types.

---

## 5. Phase 3 — Fine-Grained Retrieval (PageIndex / Vectorless Layer)

> **Blocked on the §8 data-sovereignty decision.** Do not start until resolved.

The `pageindex_tree_search` node, reached via a conditional edge for structured/complex queries.

- **Tree construction** per document (offline indexing): hierarchical parse → JSON tree → metadata
  (page/section IDs) → per-node LLM summaries.
- **Fallback cascade** for TOC extraction: (1) with page numbers → (2) without → (3) pure LLM segmentation.
- **Reasoning-based tree search**: LLM walks Root→Section→Page so footnotes stay attached to their tables.
  Expressible as a bounded recursive sub-graph.
- **Routing**: only structured/complex queries reach this node (it costs *seconds*, not ms).

**Exit criteria:** on structured-doc Q&A, tree search measurably reduces context-fragmentation errors
vs. chunk-based retrieval — and the accuracy gain justifies the latency.

---

## 6. Phase 4 — Corrective RAG (CRAG) & Quality Gate

The cyclic core of the graph.

- **`retrieval_evaluator` node + 3-way conditional edge** (lightweight classifier scoring retrieved docs):
  - High → `generate`. Medium → `secondary_retrieve` (web/secondary fallback, loops back). Low → discard +
    restart retrieval or `ask_user`.
- **`self_rag_reflect` node**: generator critiques output for faithfulness + relevance via reflection
  tokens; a conditional edge loops back to `generate` if unfaithful. **Max-iteration guard in state.**
- Always emit **citations** (page/section/node IDs) — the checkpointer's state trace doubles as the audit log.

**Exit criteria:** faithfulness up, hallucination rate down on eval, with bounded added latency and a
hard cap on retry cycles.

---

## 7. Phase 5 — Optimization & Scale *(only after correctness is proven)*

- **LLMLingua** context compression (a node before `generate`) to cut token tax and "Lost in the Middle".
- **Proxy-Pointer RAG**: embed structural metadata into the vector index for vectorless-grade accuracy at vector speed.
- **PageIndex File System** layer for navigation across enterprise-scale corpora.
- Load/latency testing against the p95 + cost budgets from Phase 0; consider LangGraph async nodes /
  parallel fan-out for multi-query and multi-doc.

---

## 8. Open decisions (need answers before/at the marked phases)

| # | Decision | Why it matters | Needed by |
|---|----------|----------------|-----------|
| 1 | **Regulated data? (HIPAA/SOC2/finance)** | If yes, PageIndex→OpenAI baseline is **disqualified** without a BAA; we must use a local/self-hosted model or a BAA-covered API. | Phase 3 |
| 2 | **LLM provider** | Directive mandates GPT-4o. Alternatives (incl. Claude, or local models) may be required for #1 or for cost. LangGraph is provider-agnostic, so this is swappable. | Phase 1 |
| 3 | **Vector DB: Pinecone vs Qdrant** | Managed vs self-host; affects sovereignty + ops. | Phase 2 |
| 4 | **Corpus + eval set source** | Everything in Phase 0 depends on representative docs. | Phase 0 |

---

## 9. Suggested sequencing (dependency-ordered)

1. **Phase 0** — scaffold + `RAGState` + eval harness + corpus. *(unblocks everything)*
2. **Phase 2** — hybrid search + RRF + reranker (`hybrid_retrieve`). *(biggest accuracy ROI; ship a working RAG)*
3. **Phase 1** — router + query transforms (entry nodes). *(cut latency/cost on the now-working system)*
4. **Resolve §8 decisions #1–#2.**
5. **Phase 3** — PageIndex tree search node.
6. **Phase 4** — CRAG + Self-RAG cyclic gate.
7. **Phase 5** — compression, Proxy-Pointer, scale.

> Deliberate reorder vs. the directive: build the **measurable workhorse (Phase 2) before the adaptive
> front-end (Phase 1)** — you can't tune a router for a retriever that doesn't exist yet. LangGraph makes
> this incremental: start with a 2-node graph and add/rewire nodes as each phase lands.

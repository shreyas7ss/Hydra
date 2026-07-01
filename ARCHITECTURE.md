# Hydra — Architecture & Progress Map

The system is one LangGraph state machine over a shared `RAGState`. Solid arrows are the
**currently wired** flow; dashed arrows/nodes are **planned** (Phases 3–5). Colour marks
build status.

```mermaid
flowchart TD
    Q([User query]) --> RI["route_intent — Intent Gate"]

    RI -->|direct| DL["direct_lookup — BM25 fast path"]
    RI -->|complex| TQ["transform_query — MultiQuery / Decompose / HyDE"]

    TQ --> HR["hybrid_retrieve — dense HNSW + BM25, RRF k=60, rerank"]

    %% ---- current terminus ----
    DL --> OUT(["Candidates + audit trail  (current END)"])
    HR --> OUT

    %% ---- Phase 3: Vectorless / PageIndex ----
    subgraph P3 ["Phase 3 · Vectorless / PageIndex"]
      PI["pageindex_tree_search — Root to Section to Page"]
    end
    HR -.->|structured docs| PI

    %% ---- Phase 4: Corrective RAG ----
    subgraph P4 ["Phase 4 · Corrective RAG quality gate"]
      RE["retrieval_evaluator — high / medium / low"]
      SEC["secondary_retrieve — web / fallback"]
      GEN["generate"]
      SR["self_rag_reflect — faithfulness + relevance"]
    end
    HR -.-> RE
    PI -.-> RE
    RE -.->|high| GEN
    RE -.->|medium| SEC
    RE -.->|low| ASK["restart / ask_user"]
    SEC -.-> RE
    GEN -.-> SR
    SR -.->|unfaithful, loop| GEN
    SR -.->|ok| ANS(["Answer + citations  (page / section / node IDs)"])

    %% ---- Phase 5: Optimization ----
    subgraph P5 ["Phase 5 · Optimization & Scale"]
      LL["LLMLingua compression"]
      PP["Proxy-Pointer RAG"]
    end
    RE -.-> LL -.-> GEN
    PP -.-> HR

    %% ---- Phase 0: cross-cutting eval ----
    EVAL["Phase 0 · Eval harness — HitRate / MRR / latency / tokens"]
    EVAL -.observes.-> RI
    EVAL -.observes.-> HR

    classDef done fill:#d4f4dd,stroke:#2e7d32,color:#1b5e20;
    classDef pending fill:#f2f2f2,stroke:#9e9e9e,color:#616161,stroke-dasharray: 5 5;
    classDef io fill:#e3f0ff,stroke:#1565c0,color:#0d47a1;

    class RI,DL,TQ,HR,EVAL done;
    class PI,RE,SEC,GEN,SR,ASK,LL,PP pending;
    class Q,OUT,ANS io;
```

**Legend:** green = built & tested · grey dashed = planned · blue = input/output.

## Completion status

| Phase | Scope | Status |
|-------|-------|--------|
| **Phase 0** | Eval harness — HitRate/MRR/recall/precision@k, latency p50/p95, token accounting, CI gate | ✅ **Done** |
| **Phase 1** | `route_intent` (gate) + `transform_query` (Multi-Query / Decompose / HyDE) | ✅ **Done** |
| **Phase 2** | `hybrid_retrieve` (dense HNSW + BM25 → RRF k=60 → rerank) + `direct_lookup` | ✅ **Done** |
| **Phase 3** | `pageindex_tree_search` — vectorless tree reasoning | ⬜ Planned *(gated on §8 data-sovereignty decision)* |
| **Phase 4** | CRAG `retrieval_evaluator` + `generate` + `self_rag_reflect` cycle | ⬜ Planned |
| **Phase 5** | LLMLingua compression + Proxy-Pointer RAG | ⬜ Planned |

**We are here:** the graph runs end-to-end through **routing → query transformation →
hybrid retrieval**, returning ranked candidates with a page/section audit trail, all scored
by the Phase 0 harness. Everything past `hybrid_retrieve` (the dashed region) is not yet built —
the graph currently terminates at retrieved candidates; generation/citations arrive in Phase 4.

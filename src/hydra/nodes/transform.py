"""Phase 1 — Query Rewriting & Transformation.

Runs the enabled transforms to bridge the query/document semantic gap, then assembles
a deduped fan-out of search queries for the (Phase 2) retriever:

* Multi-Query Expansion — parallel paraphrases to enrich the candidate pool.
* Sub-Query Decomposition — split a multi-hop question into sequential sub-questions.
* HyDE — a hypothetical answer used downstream as the embedding seed (answer-to-answer
  similarity), so it is carried in state rather than added to the search set here.

Each transform is independently flag-gated so the eval harness can A/B them.
"""

from __future__ import annotations

from hydra.config import Settings
from hydra.llm import LLMClient, parse_str_list

MULTI_QUERY_SYSTEM = """You rewrite a search query into several diverse paraphrases that
preserve intent but vary wording, specificity, and phrasing, to widen retrieval recall.
Return ONLY a JSON array of strings (do not include the original verbatim).
TASK: multi_query"""

DECOMPOSE_SYSTEM = """You decompose a complex, multi-hop question into the minimal set of
simpler sequential sub-questions that, answered together, answer the original. If the
question is already atomic, return a single-element array.
Return ONLY a JSON array of strings.
TASK: decompose"""

HYDE_SYSTEM = """You write a short, plausible hypothetical passage that would directly
answer the user's question, as if excerpted from an authoritative source document. Do not
hedge or mention that it is hypothetical. 2-4 sentences.
TASK: hyde"""


def _dedup_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip().lower()
        if item.strip() and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def make_transform_query(llm: LLMClient, settings: Settings):
    """Build the ``transform_query`` node bound to an LLM + settings."""

    def transform_query(state: dict) -> dict:
        query = state["query"]
        expanded: list[str] = []
        sub_queries: list[str] = []
        hyde_doc = ""

        if settings.enable_multi_query:
            raw = llm.complete(system=MULTI_QUERY_SYSTEM, user=query)
            expanded = parse_str_list(raw)[: settings.multi_query_count]

        if settings.enable_decomposition:
            raw = llm.complete(system=DECOMPOSE_SYSTEM, user=query)
            sub_queries = parse_str_list(raw)

        if settings.enable_hyde:
            hyde_doc = llm.complete(system=HYDE_SYSTEM, user=query).strip()

        # Final fan-out: original query first, then expansions and sub-questions.
        search_queries = _dedup_keep_order([query, *expanded, *sub_queries])

        detail = (
            f"{len(expanded)} expansions, {len(sub_queries)} sub-queries, "
            f"hyde={'yes' if hyde_doc else 'no'} -> {len(search_queries)} search queries"
        )
        return {
            "expanded_queries": expanded,
            "sub_queries": sub_queries,
            "hyde_doc": hyde_doc,
            "search_queries": search_queries,
            "trace": [{"node": "transform_query", "detail": detail}],
        }

    return transform_query

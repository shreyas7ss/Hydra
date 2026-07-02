"""Command-line entry point: run a query through the graph.

    hydra "what was net revenue in 2023?"               # configured providers, stub retrieval
    hydra --demo "compare 2022 and 2023 margins"        # offline: demo LLM + sample corpus retrieval
    hydra --demo --corpus docs.jsonl "clause 7.2"       # offline LLM over your own corpus

A JSONL corpus file has one object per line: {"id": "...", "text": "...", "metadata": {...}}.
"""

from __future__ import annotations

import argparse
import json
import sys

from hydra.config import Settings
from hydra.graph import run_query
from hydra.llm import build_llm


def _load_corpus(path: str):
    from hydra.retrieval.documents import Document

    docs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            docs.append(
                Document(id=str(obj["id"]), text=obj["text"], metadata=obj.get("metadata", {}))
            )
    return docs


def _build_retriever(settings: Settings, *, demo: bool, corpus_path: str | None):
    """Build a retriever when we have a corpus; otherwise return None (stub retrieval)."""
    if corpus_path:
        docs = _load_corpus(corpus_path)
    elif demo:
        from hydra.sample_data import sample_documents

        docs = sample_documents()
    else:
        return None

    from hydra.retrieval import HybridRetriever

    return HybridRetriever.from_settings(docs, settings=settings, demo=demo)


def _print_result(query: str, state: dict) -> None:
    print(f"\nQuery:  {query}")
    print(f"Intent: {state.get('intent')} (confidence={state.get('intent_confidence')})")
    if state.get("intent_reasoning"):
        print(f"Reason: {state['intent_reasoning']}")
    print(f"Path:   {state.get('retrieval_path')}")
    if state.get("retrieval_confidence"):
        print(f"CRAG:   {state['retrieval_confidence']} confidence "
              f"(score={state.get('retrieval_score', 0):.2f})")

    if state.get("answer"):
        print(f"\nAnswer: {state['answer']}")
    if state.get("citations"):
        cites = ", ".join(
            f"{c.get('source', '?')} p.{c.get('page', '?')} sec.{c.get('section', '?')}"
            for c in state["citations"]
        )
        print(f"Cited:  {cites}")
    if state.get("reflection"):
        r = state["reflection"]
        print(f"Self-RAG: faithful={r.get('faithful')} relevant={r.get('relevant')}")

    if state.get("expanded_queries"):
        print("\nMulti-query expansions:")
        for q in state["expanded_queries"]:
            print(f"  - {q}")
    if state.get("sub_queries"):
        print("\nSub-queries:")
        for q in state["sub_queries"]:
            print(f"  - {q}")
    if state.get("hyde_doc"):
        print(f"\nHyDE seed: {state['hyde_doc']}")

    candidates = state.get("candidates") or []
    if candidates:
        print(f"\nRetrieved candidates ({len(candidates)}):")
        for i, c in enumerate(candidates, 1):
            meta = c.get("metadata", {})
            loc = f"{meta.get('source', '?')} p.{meta.get('page', '?')} sec.{meta.get('section', '?')}"
            print(f"  {i}. [{c['score']:.3f}] {c['id']} ({loc})")
            print(f"     {c['text'][:110]}{'...' if len(c['text']) > 110 else ''}")
            if c.get("sources"):
                print(f"     via: {', '.join(c['sources'])}")

    print("\nTrace (audit path):")
    for step in state.get("trace", []):
        print(f"  {step['node']}: {step['detail']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hydra", description="Hydra Hybrid Adaptive RAG — Phase 1 front-end + Phase 2 retrieval"
    )
    parser.add_argument("query", help="the user query to route, transform, and retrieve for")
    parser.add_argument("--demo", action="store_true",
                        help="offline mode: demo LLM + hashing embedder + lexical reranker")
    parser.add_argument("--corpus", metavar="PATH",
                        help="JSONL corpus to index ({id, text, metadata} per line)")
    parser.add_argument("--json", action="store_true", help="print the final state as JSON")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    try:
        llm = build_llm(settings, demo=args.demo)
        retriever = _build_retriever(settings, demo=args.demo, corpus_path=args.corpus)
    except (RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    state = run_query(args.query, llm=llm, settings=settings, retriever=retriever)

    if args.json:
        print(json.dumps(state, indent=2, default=str))
    else:
        _print_result(args.query, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Command-line entry point: run a query through the Phase 1 front-end.

    hydra "what was net revenue in 2023?"        # uses configured provider
    hydra --demo "compare 2022 and 2023 margins" # offline, no API key needed
"""

from __future__ import annotations

import argparse
import json
import sys

from hydra.config import Settings
from hydra.graph import run_query
from hydra.llm import build_llm


def _print_result(query: str, state: dict) -> None:
    print(f"\nQuery:  {query}")
    print(f"Intent: {state.get('intent')} "
          f"(confidence={state.get('intent_confidence')})")
    if state.get("intent_reasoning"):
        print(f"Reason: {state['intent_reasoning']}")
    print(f"Path:   {state.get('retrieval_path')}")

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
    if state.get("search_queries"):
        print(f"\nSearch fan-out ({len(state['search_queries'])}):")
        for q in state["search_queries"]:
            print(f"  - {q}")

    print("\nTrace (audit path):")
    for step in state.get("trace", []):
        print(f"  {step['node']}: {step['detail']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hydra", description="Hydra Hybrid Adaptive RAG — Phase 1 front-end"
    )
    parser.add_argument("query", help="the user query to route + transform")
    parser.add_argument("--demo", action="store_true",
                        help="use the offline demo LLM (no API key required)")
    parser.add_argument("--json", action="store_true",
                        help="print the final state as JSON")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    try:
        llm = build_llm(settings, demo=args.demo)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    state = run_query(args.query, llm=llm, settings=settings)

    if args.json:
        print(json.dumps(state, indent=2, default=str))
    else:
        _print_result(args.query, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

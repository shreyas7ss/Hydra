"""`hydra-eval` — run the evaluation harness over a golden set.

    hydra-eval --demo                       # offline: demo LLM + sample corpus + built-in golden set
    hydra-eval --demo --dataset gold.jsonl --corpus docs.jsonl
    hydra-eval --json                       # machine-readable output

Exit code is non-zero if hit_rate@3 falls below --min-hit-rate (a CI regression gate).
"""

from __future__ import annotations

import argparse
import json
import sys

from hydra.config import Settings
from hydra.eval import evaluate, load_dataset, sample_golden_set
from hydra.eval.report import render_text
from hydra.llm import build_llm


def _build_retriever(settings: Settings, *, demo: bool, corpus_path: str | None):
    if corpus_path:
        from hydra.cli import _load_corpus

        docs = _load_corpus(corpus_path)
    else:
        from hydra.sample_data import sample_documents

        docs = sample_documents()
    from hydra.retrieval import HybridRetriever

    return HybridRetriever.from_settings(docs, settings=settings, demo=demo)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hydra-eval", description="Hydra evaluation harness")
    parser.add_argument("--demo", action="store_true",
                        help="offline mode (demo LLM + hashing embedder + lexical reranker)")
    parser.add_argument("--dataset", metavar="PATH",
                        help="golden JSONL; defaults to the built-in sample golden set")
    parser.add_argument("--corpus", metavar="PATH",
                        help="corpus JSONL; defaults to the built-in sample corpus")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("--min-hit-rate", type=float, default=0.0,
                        help="fail (exit 1) if mean hit_rate@3 is below this (CI gate)")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    try:
        llm = build_llm(settings, demo=args.demo)
        retriever = _build_retriever(settings, demo=args.demo, corpus_path=args.corpus)
    except (RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    dataset = load_dataset(args.dataset) if args.dataset else sample_golden_set()
    report = evaluate(dataset, llm=llm, settings=settings, retriever=retriever)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(render_text(report, label="demo" if args.demo else "configured"))

    hit_rate_3 = report.aggregate.get("hit_rate@3", 0.0)
    if hit_rate_3 < args.min_hit_rate:
        print(f"\nFAIL: hit_rate@3={hit_rate_3:.3f} < min {args.min_hit_rate:.3f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Human-readable rendering of an EvalReport."""

from __future__ import annotations

from hydra.eval.runner import EvalReport


def render_text(report: EvalReport, *, label: str = "") -> str:
    agg = report.aggregate
    lat = report.latency
    llm = report.llm
    lines: list[str] = []

    header = f"Hydra eval - {len(report.per_query)} queries"
    if label:
        header += f" ({label})"
    lines.append(header)
    lines.append("=" * len(header))

    lines.append("\nRetrieval quality (mean):")
    for k in report.k_values:
        lines.append(
            f"  @{k:<2}  hit_rate={agg.get(f'hit_rate@{k}', 0):.3f}"
            f"  recall={agg.get(f'recall@{k}', 0):.3f}"
            f"  precision={agg.get(f'precision@{k}', 0):.3f}"
        )
    lines.append(f"  MRR={agg.get('mrr', 0):.3f}")
    if "evidence_page_recall" in agg:
        lines.append(f"  evidence_page_recall={agg['evidence_page_recall']:.3f}")

    lines.append("\nOps / Retrieval Tax:")
    lines.append(
        f"  latency  mean={lat['mean'] * 1000:.1f}ms"
        f"  p50={lat['p50'] * 1000:.1f}ms  p95={lat['p95'] * 1000:.1f}ms"
    )
    lines.append(
        f"  llm      {int(llm['calls'])} calls, ~{int(llm['approx_tokens'])} tokens"
        f"  ({llm['calls_per_query']:.1f} calls/query, "
        f"~{llm['approx_tokens_per_query']:.0f} tokens/query)"
    )

    lines.append("\nGeneration quality:")
    if report.generation_active:
        gen_line = (
            f"  faithfulness={agg.get('faithfulness', 0):.3f}"
            f"  answer_relevance={agg.get('answer_relevance', 0):.3f}"
        )
        if "correctness" in agg:
            gen_line += f"  correctness={agg['correctness']:.3f}"
        lines.append(gen_line)
    else:
        lines.append("  n/a - no generator yet (activates in Phase 4)")

    lines.append("\nPer-query:")
    lines.append(f"  {'hit@3':>6} {'mrr':>5} {'lat(ms)':>8}  intent    query")
    for r in report.per_query:
        lines.append(
            f"  {r.metrics.get('hit_rate@3', 0):>6.0f}"
            f" {r.metrics.get('mrr', 0):>5.2f}"
            f" {r.latency_s * 1000:>8.1f}"
            f"  {(r.intent or '?'):<8}  {r.example.query}"
        )

    return "\n".join(lines)

"""Phase 0 — Foundation & Evaluation Harness.

Measure before optimizing. This package scores the pipeline so every later phase is
judged, not assumed:

* Retrieval quality — Hit Rate, MRR, recall@k, precision@k (live now).
* Ops / "Retrieval Tax" — latency percentiles + LLM call/token accounting (live now).
* Generation quality — faithfulness / answer-relevance proxies (library functions;
  they activate once Phase 4 produces an answer).
"""

from hydra.eval.dataset import EvalExample, load_dataset, sample_golden_set
from hydra.eval.metrics import retrieval_metrics
from hydra.eval.runner import EvalReport, evaluate

__all__ = [
    "EvalExample",
    "load_dataset",
    "sample_golden_set",
    "retrieval_metrics",
    "evaluate",
    "EvalReport",
]

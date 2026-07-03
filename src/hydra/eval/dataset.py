"""Golden evaluation set.

Each example pairs a query with the doc IDs that *should* be retrieved (the known
source spans the plan requires) and, optionally, a reference answer for generation
metrics later. The built-in set targets the bundled sample corpus so `hydra-eval --demo`
runs with zero setup.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalExample:
    query: str
    relevant_ids: list[str] = field(default_factory=list)  # gold doc/node IDs
    answer: str | None = None        # gold answer (enables the correctness judge)
    evidence_pages: list[int] = field(default_factory=list)  # gold physical pages
    metadata: dict = field(default_factory=dict)


# Golden set over hydra.sample_data.SAMPLE_DOCUMENTS.
_GOLDEN: list[EvalExample] = [
    EvalExample("what was the operating margin in 2023?", ["fin-2023-margin"]),
    EvalExample("compare operating margin between 2022 and 2023",
                ["fin-2023-margin", "fin-2022-revenue"]),
    EvalExample("what was total net revenue in fiscal 2023?", ["fin-2023-revenue"]),
    EvalExample("what net revenue did the company report in 2022?", ["fin-2022-revenue"]),
    EvalExample("what does clause 7.2 limitation of liability say?", ["legal-clause-7-2"]),
    EvalExample("how can the agreement be terminated for breach?", ["legal-clause-9-1"]),
    EvalExample("how much paid time off do employees accrue?", ["hr-pto-policy"]),
    EvalExample("how does the company recognize revenue?", ["footnote-revenue-recognition"]),
    EvalExample("what supply chain risks does the company face?", ["risk-supply-chain"]),
]


def sample_golden_set() -> list[EvalExample]:
    return list(_GOLDEN)


def load_dataset(path: str) -> list[EvalExample]:
    """Load a JSONL eval set: {"query", "relevant_ids", "answer"?, "metadata"?} per line."""
    examples: list[EvalExample] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            examples.append(
                EvalExample(
                    query=obj["query"],
                    relevant_ids=list(obj.get("relevant_ids", [])),
                    answer=obj.get("answer"),
                    evidence_pages=[int(p) for p in obj.get("evidence_pages", [])],
                    metadata=obj.get("metadata", {}),
                )
            )
    return examples

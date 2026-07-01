"""A tiny built-in corpus so `hydra --demo` can show real Phase 2 retrieval offline.

Finance/legal-flavoured snippets with page/section metadata (mirroring the structured
documents the directive targets) — enough to demonstrate hybrid retrieval + fusion +
reranking and the audit trail, not a benchmark set.
"""

from __future__ import annotations

from hydra.retrieval.documents import Document

SAMPLE_DOCUMENTS: list[Document] = [
    Document(
        id="fin-2023-revenue",
        text="In fiscal year 2023, the company reported total net revenue of $1.24 billion, "
        "up from $1.05 billion in 2022, driven by growth in the cloud segment.",
        metadata={"source": "10-K 2023", "page": 42, "section": "Results of Operations"},
    ),
    Document(
        id="fin-2022-revenue",
        text="For fiscal year 2022, total net revenue was $1.05 billion, with operating "
        "margin of 18.2 percent.",
        metadata={"source": "10-K 2022", "page": 39, "section": "Results of Operations"},
    ),
    Document(
        id="fin-2023-margin",
        text="Operating margin for 2023 expanded to 21.5 percent from 18.2 percent in 2022 "
        "as gross margin improved and operating expenses grew slower than revenue.",
        metadata={"source": "10-K 2023", "page": 43, "section": "Margins"},
    ),
    Document(
        id="legal-clause-7-2",
        text="Clause 7.2 (Limitation of Liability): neither party shall be liable for indirect, "
        "incidental, or consequential damages, and total aggregate liability is capped at "
        "the fees paid in the preceding twelve months.",
        metadata={"source": "MSA", "page": 11, "section": "7.2"},
    ),
    Document(
        id="legal-clause-9-1",
        text="Clause 9.1 (Termination): either party may terminate this agreement for material "
        "breach upon thirty days written notice if the breach remains uncured.",
        metadata={"source": "MSA", "page": 14, "section": "9.1"},
    ),
    Document(
        id="risk-supply-chain",
        text="The company faces supply-chain risk concentrated in a small number of component "
        "suppliers; disruption could materially affect product availability and revenue.",
        metadata={"source": "10-K 2023", "page": 21, "section": "Risk Factors"},
    ),
    Document(
        id="footnote-revenue-recognition",
        text="Footnote 3: revenue is recognized when control of the promised goods or services "
        "transfers to the customer, net of estimated returns and allowances.",
        metadata={"source": "10-K 2023", "page": 58, "section": "Notes"},
    ),
    Document(
        id="hr-pto-policy",
        text="Employees accrue paid time off at a rate of 1.5 days per month, up to a maximum "
        "balance of 30 days, after which accrual pauses.",
        metadata={"source": "Employee Handbook", "page": 7, "section": "Time Off"},
    ),
]


def sample_documents() -> list[Document]:
    return list(SAMPLE_DOCUMENTS)

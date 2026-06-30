"""Hydra — Hybrid Adaptive RAG (vector + vectorless) orchestrated with LangGraph.

Phase 1 (Query Intelligence Layer) is implemented here: an intent-classification
gate plus query transformation (multi-query expansion, sub-query decomposition,
HyDE). Downstream retrieval nodes are thin stubs until Phases 2/3 land.
"""

from hydra.state import RAGState

__all__ = ["RAGState"]
__version__ = "0.1.0"

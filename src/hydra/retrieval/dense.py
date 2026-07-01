"""Dense vector index over Qdrant with HNSW (plan §8 decision #3 = Qdrant).

Uses Qdrant's local in-memory mode (``location=":memory:"``) so it runs with no server
and no network — the same code points at a real Qdrant cluster by swapping the client
constructor (``url=...``). HNSW params are set explicitly to honour the directive's
"sub-millisecond similarity search" coarse-filter requirement at scale.
"""

from __future__ import annotations

from hydra.retrieval.documents import Document
from hydra.retrieval.embeddings import Embedder

_COLLECTION = "hydra_corpus"


class DenseIndex:
    def __init__(self, embedder: Embedder, *, hnsw_m: int = 16, hnsw_ef_construct: int = 100) -> None:
        from qdrant_client import QdrantClient

        self.embedder = embedder
        self.client = QdrantClient(location=":memory:")
        self._hnsw_m = hnsw_m
        self._hnsw_ef_construct = hnsw_ef_construct
        self._id_to_doc: dict[int, str] = {}

    def build(self, documents: list[Document]) -> None:
        from qdrant_client import models

        if self.client.collection_exists(_COLLECTION):
            self.client.delete_collection(_COLLECTION)
        self.client.create_collection(
            collection_name=_COLLECTION,
            vectors_config=models.VectorParams(
                size=self.embedder.dim,
                distance=models.Distance.COSINE,
            ),
            hnsw_config=models.HnswConfigDiff(
                m=self._hnsw_m,
                ef_construct=self._hnsw_ef_construct,
            ),
        )
        vectors = self.embedder.embed([d.text for d in documents])
        self._id_to_doc = {i: d.id for i, d in enumerate(documents)}
        points = [
            models.PointStruct(id=i, vector=vec, payload={"doc_id": d.id})
            for i, (d, vec) in enumerate(zip(documents, vectors))
        ]
        if points:
            self.client.upsert(collection_name=_COLLECTION, points=points)

    def search(self, query_text: str, top_k: int) -> list[tuple[str, float]]:
        if not self._id_to_doc:
            return []
        vector = self.embedder.embed([query_text])[0]
        response = self.client.query_points(
            collection_name=_COLLECTION,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
        return [(point.payload["doc_id"], float(point.score)) for point in response.points]

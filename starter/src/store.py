"""Persistent local Qdrant adapter; the optional pgvector setup is not used."""
import math
import uuid
from typing import Any
from qdrant_client import QdrantClient, models
from .models import Chunk


class QdrantStore:
    def __init__(self, storage_path: str, dimensions: int) -> None:
        """Open the official client's local persistent backend, not an in-memory substitute."""
        if type(dimensions) is not int or dimensions < 1:
            raise ValueError('dimensions must be a positive integer')
        self.dimensions = dimensions
        self.client = QdrantClient(path=storage_path)

    def create_schema(self) -> None:
        """Validate backend access; strategy collections are initialized at upsert."""
        self.client.get_collections()

    def _collection(self, strategy: str) -> str:
        if strategy not in ('A', 'B', 'C'):
            raise ValueError('Unknown strategy')
        return 'day06_' + strategy.lower()

    def _vector(self, vector: list[float]) -> list[float]:
        if len(vector) != self.dimensions or not all(math.isfinite(float(v)) for v in vector) or not any(vector):
            raise ValueError('Vector has an invalid dimension, non-finite value or zero norm')
        return [float(v) for v in vector]

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Replace one strategy's complete corpus snapshot using stable IDs.

        When re-ingesting changed source documents, remove obsolete chunks too.
        """
        if len(chunks) != len(vectors) or len({c.chunk_id for c in chunks}) != len(chunks):
            raise ValueError('Chunk/vector counts differ or chunk IDs are duplicated')
        vectors = [self._vector(v) for v in vectors]  # Validate before changing stored data.
        name = self._collection(strategy)
        if not self.client.collection_exists(name):
            self.client.create_collection(name, vectors_config=models.VectorParams(size=self.dimensions, distance=models.Distance.COSINE))
        params = self.client.get_collection(name).config.params.vectors
        if not isinstance(params, models.VectorParams) or params.size != self.dimensions or params.distance != models.Distance.COSINE:
            raise ValueError('Existing collection is incompatible with this embedding space')
        old, offset = set(), None
        while True:
            records, offset = self.client.scroll(name, limit=256, offset=offset, with_payload=False, with_vectors=False)
            old.update(record.id for record in records)
            if offset is None:
                break
        points = [models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.chunk_id)), vector=v,
                                    payload={**c.metadata, 'document_id': c.document_id, 'source': c.source,
                                             'page': c.page, 'chunk_id': c.chunk_id, 'text': c.text})
                  for c, v in zip(chunks, vectors)]
        for start in range(0, len(points), 128):
            self.client.upsert(name, points=points[start:start + 128], wait=True)
        obsolete = old - {p.id for p in points}
        if obsolete:
            self.client.delete(name, points_selector=models.PointIdsList(points=list(obsolete)), wait=True)

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Filter permitted metadata in Qdrant, rank by cosine similarity, return top-k.

        Return chunk ID, text, score and metadata. Parameterize user-supplied values.
        """
        if type(top_k) is not int or not 1 <= top_k <= 100:
            raise ValueError('top_k must be between 1 and 100')
        conditions = []
        for key, value in (filters or {}).items():
            if key not in ('year', 'document_type') or type(value) not in (str, int):
                raise ValueError('Only trusted year/document_type filters are allowed, never gold source or answer')
            conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
        hits = self.client.query_points(self._collection(strategy), query=self._vector(query_vector), limit=top_k,
                                        query_filter=models.Filter(must=conditions) if conditions else None,
                                        with_payload=True).points
        return [{'chunk_id': h.payload['chunk_id'], 'text': h.payload['text'], 'score': float(h.score),
                 'metadata': {k: v for k, v in h.payload.items() if k not in ('chunk_id', 'text')}} for h in hits]

    def count(self, strategy: str) -> int:
        return self.client.count(self._collection(strategy), exact=True).count

    def close(self) -> None:
        self.client.close()

"""Vector database service backed by a local Qdrant instance."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http import models


class QdrantService:
    """Service wrapper for interacting with Qdrant collections."""

    def __init__(self, base_storage_path: str = "data/bases") -> None:
        self._base_storage_path = Path(base_storage_path)
        self._base_storage_path.mkdir(parents=True, exist_ok=True)
        self._clients: Dict[str, QdrantClient] = {}

    def _get_client(self, base_name: str) -> QdrantClient:
        if base_name not in self._clients:
            storage_path = self._base_storage_path / base_name / "qdrant"
            storage_path.mkdir(parents=True, exist_ok=True)
            self._clients[base_name] = QdrantClient(path=str(storage_path))
        return self._clients[base_name]

    def create_collection(self, base_name: str) -> None:
        """Create or recreate a collection with the expected vector size."""

        client = self._get_client(base_name)
        client.recreate_collection(
            collection_name=base_name,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
        )

    def add_documents(
        self,
        base_name: str,
        chunks: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadata: Sequence[Optional[Dict[str, Any]]],
    ) -> List[str]:
        """Add text chunks with embeddings and metadata to a collection."""

        client = self._get_client(base_name)

        if not (len(chunks) == len(embeddings) == len(metadata)):
            raise ValueError("chunks, embeddings, and metadata must be the same length")

        points: List[models.PointStruct] = []
        document_ids: List[str] = []
        for chunk, embedding, meta in zip(chunks, embeddings, metadata):
            document_id = (meta or {}).get("id") or str(uuid4())
            payload = {"text": chunk}
            if meta:
                payload.update(meta)
            points.append(
                models.PointStruct(
                    id=document_id,
                    vector=list(map(float, embedding)),
                    payload=payload,
                )
            )
            document_ids.append(document_id)

        client.upsert(collection_name=base_name, points=points)
        return document_ids

    def search(
        self,
        query_embedding: Sequence[float],
        base_name: str,
        limit: int = 3,
        filter: Optional[models.Filter] = None,
    ) -> List[models.ScoredPoint]:
        """Search similar vectors for the provided embedding."""

        client = self._get_client(base_name)
        return client.search(
            collection_name=base_name,
            query_vector=list(map(float, query_embedding)),
            limit=limit,
            query_filter=filter,
        )

    def delete_document(self, document_id: str, base_name: str) -> None:
        """Delete a document from a collection by its identifier."""

        client = self._get_client(base_name)
        client.delete(
            collection_name=base_name,
            points_selector=models.PointIdsList(points=[document_id]),
        )


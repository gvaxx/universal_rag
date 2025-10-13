"""SQLite-based provider stubs for document and vector storage.

This is a minimal stub that does not implement real vector search. It is a placeholder
to demonstrate the provider API and will be expanded in future iterations.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

from .base import DocumentStore, VectorStore


class SQLiteDocumentStore(DocumentStore):
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def add_documents(self, documents: Iterable[dict]) -> None:  # pragma: no cover - stub
        pass

    def get_document(self, doc_id: str) -> Optional[dict]:  # pragma: no cover - stub
        return None


class SQLiteVectorStore(VectorStore):
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def upsert(
        self, ids: List[str], embeddings: List[List[float]], metadatas: List[dict]
    ) -> None:  # pragma: no cover - stub
        pass

    def search(self, embedding: List[float], top_k: int) -> List[dict]:  # pragma: no cover - stub
        return []



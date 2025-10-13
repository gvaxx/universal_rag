"""ABCs for VectorStore and DocumentStore."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, ids: List[str], embeddings: List[List[float]], metadatas: List[dict]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, embedding: List[float], top_k: int) -> List[dict]:
        raise NotImplementedError


class DocumentStore(ABC):
    @abstractmethod
    def add_documents(self, documents: Iterable[dict]) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_document(self, doc_id: str) -> dict | None:
        raise NotImplementedError



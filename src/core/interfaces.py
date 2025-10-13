"""Abstract interfaces for pluggable providers in the Universal RAG System."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, List, Optional, Protocol

from .models import Chunk, Document, EmbeddingResult


class SupportsClose(Protocol):
    def close(self) -> None:  # pragma: no cover - simple protocol
        ...


class EmbeddingProvider(ABC):
    """Embedding provider interface producing dense vector representations."""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """Embed a batch of texts and return an `EmbeddingResult`."""
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string into a vector."""
        raise NotImplementedError

    @abstractmethod
    def get_dimensions(self) -> int:
        """Return the dimensionality of produced embeddings."""
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:  # pragma: no cover - trivial property
        """Human-readable model name or identifier."""
        raise NotImplementedError


class LLMProvider(ABC):
    """Large Language Model provider interface."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str], **kwargs) -> str:
        """Synchronous text generation call."""
        raise NotImplementedError

    @abstractmethod
    def generate_stream(self, prompt: str, system_prompt: Optional[str], **kwargs) -> AsyncIterator[str]:
        """Asynchronous streaming generation yielding text chunks."""
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:  # pragma: no cover - trivial property
        raise NotImplementedError

    @property
    @abstractmethod
    def max_tokens(self) -> int:  # pragma: no cover - trivial property
        raise NotImplementedError


class VectorStore(ABC):
    """Interface for vector storage, search, and management."""

    @abstractmethod
    def initialize(self, collection_name: str, dimensions: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def add(self, embeddings: List[List[float]], metadata: List[Dict]) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def search(self, embedding: List[float], limit: int, filters: Optional[Dict]) -> List[Dict]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, ids: List[str]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def collection_exists(self, name: str) -> bool:
        raise NotImplementedError


class DocumentStore(ABC):
    """Interface for document and chunk persistence."""

    @abstractmethod
    def save_document(self, document: Document) -> str:
        raise NotImplementedError

    @abstractmethod
    def save_chunks(self, chunks: List[Chunk]) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def get_document(self, document_id: str) -> Optional[Document]:
        raise NotImplementedError

    @abstractmethod
    def get_chunks(self, chunk_ids: List[str]) -> List[Chunk]:
        raise NotImplementedError

    @abstractmethod
    def get_documents_by_base(self, base_name: str) -> List[Document]:
        raise NotImplementedError

    @abstractmethod
    def document_exists(self, file_hash: str, base_name: str) -> bool:
        raise NotImplementedError




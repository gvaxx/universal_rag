"""Core domain models for the Universal RAG System.

This module defines the strongly-typed dataclasses used across the system,
including documents, chunks, queries, search results, and final answers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    """Represents a source document stored/indexed by the system.

    Attributes:
        id: Stable unique identifier (UUID as string).
        filename: Original file name.
        path: Absolute or project-relative path to the file on disk.
        base_name: Logical knowledge base the document belongs to.
        total_pages: Number of pages (for paged formats like PDF).
        file_hash: Content hash for deduplication.
        file_size: File size in bytes.
        indexed_at: Timestamp when the document was indexed.
        metadata: Arbitrary metadata associated with the document.
    """

    id: str
    filename: str
    path: str
    base_name: str
    total_pages: int
    file_hash: str
    file_size: int
    indexed_at: datetime
    metadata: Dict[str, Any]


@dataclass
class Chunk:
    """A semantically coherent chunk produced from a `Document`.

    Attributes:
        id: Unique identifier (UUID as string).
        document_id: Foreign key referencing the parent `Document`.
        content: Text content of the chunk.
        page_number: Page number this chunk originates from (1-based), if applicable.
        chunk_index: Sequential index of the chunk within the document.
        start_char: Start character offset in the original document.
        end_char: End character offset in the original document.
        metadata: Arbitrary metadata for retrieval/ranking.
    """

    id: str
    document_id: str
    content: str
    page_number: int
    chunk_index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]


class QueryMode(str, Enum):
    """Supported query modes for retrieval/generation."""

    SEARCH = "SEARCH"
    QA = "QA"
    DOCUMENT = "DOCUMENT"


@dataclass
class Query:
    """A user query with mode, filters and base scoping."""

    text: str
    base_name: str
    mode: QueryMode
    filters: Optional[Dict[str, Any]]
    limit: int = 3


@dataclass
class SearchResult:
    """A single search hit with score and resolved document reference."""

    chunk: Chunk
    score: float
    document: Document


@dataclass
class Answer:
    """Final answer object produced by the RAG pipeline."""

    text: str
    sources: List[SearchResult]
    query: Query
    model_used: str
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class EmbeddingResult:
    """Embedding batch result produced by an embedding provider.

    Attributes:
        embeddings: 2D list of floats, one embedding vector per input text.
        dimensions: Length of each embedding vector.
        model_name: Name/identifier of the embedding model used.
    """

    embeddings: List[List[float]]
    dimensions: int
    model_name: str



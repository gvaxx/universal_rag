"""Indexing service stub."""

from __future__ import annotations

from typing import Iterable

from core.models import Document


class IndexingService:
    """Responsible for chunking documents and writing to stores."""

    def __init__(self) -> None:
        pass

    def index(self, documents: Iterable[Document]) -> int:  # pragma: no cover - stub
        return 0



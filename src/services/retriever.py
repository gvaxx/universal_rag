"""Retrieval service stub."""

from __future__ import annotations

from typing import List

from core.models import Chunk, Query


class RetrievalService:
    """Responsible for embedding queries and retrieving chunks."""

    def __init__(self) -> None:
        pass

    def retrieve(self, query: Query) -> List[Chunk]:  # pragma: no cover - stub
        return []



"""RAG pipeline service stub."""

from __future__ import annotations

from core.models import Answer, Query


class RAGService:
    """Orchestrates retrieval and generation to produce answers."""

    def __init__(self) -> None:
        pass

    def answer(self, query: Query) -> Answer:  # pragma: no cover - stub
        return Answer(text="", references=[])



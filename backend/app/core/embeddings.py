"""Embedding service built on top of fastembed."""

from __future__ import annotations

from typing import Dict, List

from fastembed import TextEmbedding


class EmbeddingService:
    """Service wrapper responsible for generating embeddings.

    The underlying fastembed model is cached on the class level to avoid
    repeated downloads and to keep initialization fast across the
    application lifecycle.
    """

    _models: Dict[str, TextEmbedding] = {}

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model_name = model_name
        if model_name not in self._models:
            self._models[model_name] = TextEmbedding(model_name=model_name)
        self._model = self._models[model_name]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Return embeddings for a batch of texts."""

        if not texts:
            return []

        embeddings = self._model.embed(texts)
        return [list(map(float, vector)) for vector in embeddings]

    def embed_query(self, query: str) -> List[float]:
        """Return an embedding for a single query string."""

        embeddings = self.embed_texts([query])
        return embeddings[0] if embeddings else []


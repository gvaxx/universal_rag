"""ABC for embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        raise NotImplementedError

    @abstractmethod
    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        raise NotImplementedError



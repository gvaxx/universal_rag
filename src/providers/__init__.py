"""Provider registry for plugin-based architecture.

This registry supports registering and retrieving providers of different kinds
(`embeddings`, `llm`, `storage.vector`, `storage.document`). It also exposes a
decorator to auto-register providers upon import for convenient plugin loading.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Type, TypeVar

T = TypeVar("T")


class ProviderRegistry:
    """Registry of providers organized by kind and name."""

    _embeddings: Dict[str, Type[Any]] = {}
    _llm: Dict[str, Type[Any]] = {}
    _vector_store: Dict[str, Type[Any]] = {}
    _document_store: Dict[str, Type[Any]] = {}

    # --- Register methods ---
    @classmethod
    def register_embeddings(cls, name: str, provider_cls: Type[Any]) -> None:
        cls._embeddings[name] = provider_cls

    @classmethod
    def register_llm(cls, name: str, provider_cls: Type[Any]) -> None:
        cls._llm[name] = provider_cls

    @classmethod
    def register_vector_store(cls, name: str, provider_cls: Type[Any]) -> None:
        cls._vector_store[name] = provider_cls

    @classmethod
    def register_document_store(cls, name: str, provider_cls: Type[Any]) -> None:
        cls._document_store[name] = provider_cls

    # --- Get methods with validation ---
    @classmethod
    def get_embeddings(cls, name: str) -> Type[Any]:
        if name not in cls._embeddings:
            raise KeyError(f"Embedding provider not found: {name}")
        return cls._embeddings[name]

    @classmethod
    def get_llm(cls, name: str) -> Type[Any]:
        if name not in cls._llm:
            raise KeyError(f"LLM provider not found: {name}")
        return cls._llm[name]

    @classmethod
    def get_vector_store(cls, name: str) -> Type[Any]:
        if name not in cls._vector_store:
            raise KeyError(f"Vector store provider not found: {name}")
        return cls._vector_store[name]

    @classmethod
    def get_document_store(cls, name: str) -> Type[Any]:
        if name not in cls._document_store:
            raise KeyError(f"Document store provider not found: {name}")
        return cls._document_store[name]

    # --- Introspection helpers ---
    @classmethod
    def list_providers(cls) -> Dict[str, Iterable[str]]:
        return {
            "embeddings": tuple(cls._embeddings.keys()),
            "llm": tuple(cls._llm.keys()),
            "storage.vector": tuple(cls._vector_store.keys()),
            "storage.document": tuple(cls._document_store.keys()),
        }

    # --- Decorator for auto-registration ---
    @classmethod
    def register_provider(cls, kind: str, name: str) -> Callable[[Type[T]], Type[T]]:
        """Class decorator to register a provider by kind and name upon import."""

        def decorator(provider_cls: Type[T]) -> Type[T]:
            if kind == "embeddings":
                cls.register_embeddings(name, provider_cls)
            elif kind == "llm":
                cls.register_llm(name, provider_cls)
            elif kind == "storage.vector":
                cls.register_vector_store(name, provider_cls)
            elif kind == "storage.document":
                cls.register_document_store(name, provider_cls)
            else:
                raise ValueError(f"Unknown provider kind: {kind}")
            return provider_cls

        return decorator



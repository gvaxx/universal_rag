"""Configuration module using pydantic Settings and AppConfig factories.

`Settings` reads and validates environment variables; `AppConfig` provides higher-level
helpers such as provider factories and common data paths.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from providers import ProviderRegistry


class Settings(BaseSettings):
    """Application settings loaded from environment variables (.env supported)."""

    openrouter_api_key: Optional[str] = Field(default=None, alias="OPENROUTER_API_KEY")
    default_model: str = Field(default="openai/gpt-3.5-turbo", alias="DEFAULT_MODEL")
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5", alias="EMBEDDING_MODEL")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    data_dir: str = Field(default="./data", alias="DATA_DIR")

    # RAG parameters
    chunk_size: int = Field(default=800, ge=1)
    chunk_overlap: int = Field(default=100, ge=0)

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


class AppConfig:
    """High-level configuration facade over `Settings`.

    Provides provider factories, data paths, and RAG parameters for services.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    # --- Properties delegating to settings ---
    @property
    def log_level(self) -> str:
        return self.settings.log_level

    @property
    def data_dir(self) -> str:
        return self.settings.data_dir

    @property
    def default_model(self) -> str:
        return self.settings.default_model

    @property
    def embedding_model(self) -> str:
        return self.settings.embedding_model

    @property
    def chunk_size(self) -> int:
        return self.settings.chunk_size

    @property
    def chunk_overlap(self) -> int:
        return self.settings.chunk_overlap

    # --- Data paths ---
    def get_base_path(self, base_name: str) -> str:
        return os.path.join(self.settings.data_dir, "bases", base_name)

    def get_documents_path(self, base_name: str) -> str:
        return os.path.join(self.get_base_path(base_name), "documents")

    # --- Provider factories ---
    def create_embedding_provider(self, name: str, **kwargs: Any) -> Any:
        provider_cls = ProviderRegistry.get_embeddings(name)
        return provider_cls(**kwargs)

    def create_llm_provider(self, name: str, **kwargs: Any) -> Any:
        provider_cls = ProviderRegistry.get_llm(name)
        return provider_cls(**kwargs)

    def create_vector_store(self, name: str, **kwargs: Any) -> Any:
        provider_cls = ProviderRegistry.get_vector_store(name)
        return provider_cls(**kwargs)

    def create_document_store(self, name: str, **kwargs: Any) -> Any:
        provider_cls = ProviderRegistry.get_document_store(name)
        return provider_cls(**kwargs)


def load_settings() -> Settings:
    """Load and validate settings from environment."""
    return Settings()  # pydantic reads from env and .env automatically


def build_app_config(settings: Optional[Settings] = None) -> AppConfig:
    """Build an `AppConfig` instance from settings (load if not provided)."""
    return AppConfig(settings or load_settings())



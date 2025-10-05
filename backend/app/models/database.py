"""Database models and utilities for the backend application."""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, Optional

from sqlmodel import Field, Session, SQLModel, create_engine
from sqlmodel.engine import Engine


class Document(SQLModel, table=True):
    """Represents an indexed document."""

    __tablename__ = "documents"

    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str
    path: str
    base_name: str
    total_pages: Optional[int] = None
    file_hash: str
    indexed_at: dt.datetime = Field(default_factory=lambda: dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc))


class Page(SQLModel, table=True):
    """Represents a single page of an indexed document."""

    __tablename__ = "pages"

    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="documents.id")
    page_num: int
    content: str
    char_count: int


class ChatHistory(SQLModel, table=True):
    """Stores chat history for a specific base."""

    __tablename__ = "chat_history"

    id: Optional[int] = Field(default=None, primary_key=True)
    base_name: str
    user_message: str
    assistant_response: str
    mode: str
    timestamp: dt.datetime = Field(default_factory=lambda: dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc))


_ENGINE_CACHE: Dict[str, Engine] = {}
DEFAULT_BASE_NAME = "default"
DATA_DIR = Path("data/bases")
DB_FILENAME = "cache.db"


def _database_path(base_name: str) -> Path:
    """Return the filesystem path for the database of a given base."""

    base_path = DATA_DIR / base_name
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path / DB_FILENAME


def get_engine(base_name: str = DEFAULT_BASE_NAME) -> Engine:
    """Return (and cache) the SQLModel engine for the provided base name."""

    if base_name not in _ENGINE_CACHE:
        db_path = _database_path(base_name).resolve()
        engine = create_engine(f"sqlite:///{db_path}", echo=False, connect_args={"check_same_thread": False})
        SQLModel.metadata.create_all(engine)
        _ENGINE_CACHE[base_name] = engine
    return _ENGINE_CACHE[base_name]


def get_session(base_name: str = DEFAULT_BASE_NAME) -> Session:
    """Create a new session for the provided base name."""

    engine = get_engine(base_name)
    return Session(engine)


# Ensure default database is initialised on import.
get_engine(DEFAULT_BASE_NAME)

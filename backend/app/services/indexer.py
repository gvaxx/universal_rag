"""Document indexing pipeline utilities."""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from sqlalchemy import delete
from sqlmodel import select

from ..core.chunking import TextChunker
from ..core.embeddings import EmbeddingService
from ..core.parser import DocumentParser
from ..core.vectordb import QdrantService
from ..models.database import DATA_DIR, Document, Page, get_session


ProgressCallback = Callable[[str, Optional[float]], None]


class DocumentIndexer:
    """High-level orchestrator for document indexing."""

    def __init__(
        self,
        parser: DocumentParser | None = None,
        chunker: TextChunker | None = None,
        embedding_service: EmbeddingService | None = None,
        vectordb: QdrantService | None = None,
        base_data_dir: str | Path | None = None,
        progress_callback: ProgressCallback | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.parser = parser or DocumentParser(logger=self.logger)
        self.chunker = chunker or TextChunker()
        self.embedding_service = embedding_service or EmbeddingService()
        self.vectordb = vectordb or QdrantService()
        self.base_data_dir = Path(base_data_dir) if base_data_dir else DATA_DIR
        self.progress_callback = progress_callback
        self.supported_extensions = {".pdf", ".txt", ".md"}

    def index_document(
        self,
        file_path: str | Path,
        base_name: str,
        progress_callback: ProgressCallback | None = None,
    ) -> Document:
        """Run the full indexing pipeline for a single document."""

        callback = progress_callback or self.progress_callback
        emit = self._progress_emitter(callback)

        path = Path(file_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        emit(f"Starting indexing for {path.name}", 0.0)

        file_hash = self._compute_file_hash(path)
        emit("Computed file hash", 0.05)

        session = get_session(base_name)
        try:
            document = self._get_or_create_document(session, path, base_name, file_hash)

            if document.file_hash == file_hash and document.total_pages is not None:
                # The document is up to date and already indexed.
                emit(f"No changes detected for {path.name}. Skipping.", 1.0)
                return document

            emit("Parsing document", 0.2)
            pages = self.parser.parse(path)

            emit("Clearing previous pages", 0.35)
            session.exec(delete(Page).where(Page.document_id == document.id))
            session.commit()

            document.total_pages = len(pages)
            document.indexed_at = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
            session.add(document)
            session.commit()

            emit("Storing pages", 0.45)
            self._store_pages(session, document.id, pages)
            session.commit()

            emit("Chunking content", 0.55)
            chunks, metadata = self._chunk_pages(path, document.id, pages)

            self._ensure_collection(base_name)

            if not chunks:
                emit("No content available for embeddings", 0.9)
                self._clear_document_vectors(base_name, str(path))
                emit("Indexing complete", 1.0)
                return document

            emit("Generating embeddings", 0.7)
            embeddings = self.embedding_service.embed_texts(chunks)

            emit("Updating vector store", 0.85)
            self._clear_document_vectors(base_name, str(path))
            self.vectordb.add_documents(base_name, chunks, embeddings, metadata)

            emit("Indexing complete", 1.0)
            return document
        finally:
            session.close()

    def scan_and_index_folder(
        self,
        base_name: str,
        progress_callback: ProgressCallback | None = None,
    ) -> List[Document]:
        """Scan a knowledge base folder and index new or modified files."""

        callback = progress_callback or self.progress_callback
        emit = self._progress_emitter(callback)

        documents_dir = self.base_data_dir / base_name / "documents"
        documents_dir.mkdir(parents=True, exist_ok=True)

        files: List[Path] = sorted(
            p
            for p in documents_dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.supported_extensions
        )
        total = len(files)
        indexed_documents: List[Document] = []

        emit(
            f"Found {total} file(s) to evaluate in {documents_dir}",
            0.0 if total else 1.0,
        )

        for idx, file_path in enumerate(files, start=1):
            emit(
                f"Indexing {file_path.name} ({idx}/{total})",
                (idx - 1) / total if total else None,
            )

            def nested_callback(message: str, progress: Optional[float]) -> None:
                overall_progress: Optional[float]
                if progress is not None and total:
                    overall_progress = ((idx - 1) + progress) / total
                elif total:
                    overall_progress = (idx - 1) / total
                else:
                    overall_progress = 1.0
                emit(message, overall_progress)

            try:
                document = self.index_document(
                    file_path,
                    base_name,
                    progress_callback=nested_callback,
                )
                indexed_documents.append(document)
            except Exception as exc:  # pragma: no cover - safety net for runtime issues
                self.logger.exception("Failed to index document %s", file_path)
                emit(f"Failed to index {file_path.name}: {exc}", None)

        if total:
            emit("Folder scan complete", 1.0)
        return indexed_documents

    def _progress_emitter(self, callback: ProgressCallback | None) -> ProgressCallback:
        def emit(message: str, progress: Optional[float]) -> None:
            if callback:
                callback(message, progress)

        return emit

    def _compute_file_hash(self, path: Path, chunk_size: int = 8192) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(chunk_size), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _get_or_create_document(
        self,
        session,
        path: Path,
        base_name: str,
        file_hash: str,
    ) -> Document:
        statement = select(Document).where(Document.path == str(path))
        document = session.exec(statement).first()

        if document and document.file_hash == file_hash and document.total_pages is not None:
            return document

        if not document:
            document = Document(
                filename=path.name,
                path=str(path),
                base_name=base_name,
                total_pages=None,
                file_hash=file_hash,
            )
            session.add(document)
            session.commit()
            session.refresh(document)
        else:
            document.filename = path.name
            document.path = str(path)
            document.base_name = base_name
            document.file_hash = file_hash
            document.total_pages = None
            document.indexed_at = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
            session.add(document)
            session.commit()

        return document

    def _store_pages(self, session, document_id: int, pages: Iterable[str]) -> None:
        for page_num, content in enumerate(pages, start=1):
            page = Page(
                document_id=document_id,
                page_num=page_num,
                content=content,
                char_count=len(content),
            )
            session.add(page)

    def _chunk_pages(
        self,
        path: Path,
        document_id: int,
        pages: List[str],
    ) -> tuple[List[str], List[dict]]:
        chunks: List[str] = []
        metadata: List[dict] = []
        chunk_counter = 0

        for page_num, content in enumerate(pages, start=1):
            for chunk in self.chunker.chunk(content):
                text = chunk.get("text", "").strip()
                if not text:
                    continue

                chunk_id = f"{document_id}:{page_num}:{chunk_counter}"
                metadata.append(
                    {
                        "id": chunk_id,
                        "document_id": document_id,
                        "document_path": str(path),
                        "filename": path.name,
                        "page_num": page_num,
                        "chunk_index": chunk_counter,
                        "page_chunk_index": chunk.get("index"),
                        "start_char": chunk.get("start_char"),
                        "end_char": chunk.get("end_char"),
                        "token_count": chunk.get("token_count"),
                    }
                )
                chunks.append(text)
                chunk_counter += 1

        return chunks, metadata

    def _clear_document_vectors(self, base_name: str, document_path: str) -> None:
        try:
            self.vectordb.delete_document_chunks(base_name, document_path)
        except AttributeError:
            # Fallback for vector services without the helper method.
            self.logger.debug(
                "Vector service does not expose delete_document_chunks; skipping cleanup.",
            )

    def _ensure_collection(self, base_name: str) -> None:
        client_getter = getattr(self.vectordb, "_get_client", None)
        if client_getter is None:
            return

        client = client_getter(base_name)
        try:
            client.get_collection(base_name)
        except Exception:  # pragma: no cover - depends on backend state
            self.vectordb.create_collection(base_name)


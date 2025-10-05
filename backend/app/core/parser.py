"""Document parsing utilities."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import List

from pypdf import PdfReader


class DocumentParser:
    """Parse documents into page-like chunks of text."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.last_file_hash: str | None = None

    def parse(self, file_path: str | Path) -> List[str]:
        """Parse the provided file and return a list of page texts.

        Args:
            file_path: Path to the file that should be parsed.

        Returns:
            A list of strings where each string represents a page of text.

        Raises:
            FileNotFoundError: If the provided file path does not exist.
            ValueError: If the file type is not supported.
            RuntimeError: If an unexpected error occurs during parsing.
        """

        path = Path(file_path)

        if not path.exists():
            self.logger.error("File not found: %s", path)
            raise FileNotFoundError(f"File not found: {path}")

        file_hash = self._compute_hash(path)

        try:
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                pages = self._parse_pdf(path)
            elif suffix in {".txt", ".md"}:
                pages = self._parse_text_file(path)
            else:
                self.logger.error("Unsupported file extension: %s", suffix)
                raise ValueError(f"Unsupported file type: {suffix}")
        except (FileNotFoundError, ValueError):
            raise
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.exception("Failed to parse file %s", path)
            raise RuntimeError(f"Failed to parse file {path}") from exc

        self.last_file_hash = file_hash
        self.logger.debug(
            "Parsed %s into %d page(s). Hash: %s", path, len(pages), file_hash
        )
        return pages

    def _parse_pdf(self, path: Path) -> List[str]:
        reader = PdfReader(str(path))
        pages: List[str] = []
        for index, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception:  # pragma: no cover - depends on PDF contents
                self.logger.exception("Failed to extract text from PDF page %d", index)
                text = ""

            pages.append(text.strip())

        if not pages:
            self.logger.warning("No text extracted from PDF: %s", path)
        return pages

    def _parse_text_file(self, path: Path) -> List[str]:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            content = handle.read()

        if not content:
            self.logger.warning("Empty text document: %s", path)
            return []

        return self._split_text(content)

    def _split_text(self, text: str, page_size: int = 3000) -> List[str]:
        pages: List[str] = []
        for start in range(0, len(text), page_size):
            end = start + page_size
            pages.append(text[start:end].strip())
        return pages

    def _compute_hash(self, path: Path, chunk_size: int = 8192) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(chunk_size), b""):
                digest.update(chunk)
        return digest.hexdigest()


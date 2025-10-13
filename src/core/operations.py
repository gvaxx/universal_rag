"""Pure text processing operations."""

from __future__ import annotations

from typing import Iterable, List


def split_by_chars(text: str, max_chars: int = 800, overlap: int = 100) -> List[str]:
    """Simple character-based splitter with overlap.

    Keeps it deterministic and dependency-free for the scaffold.
    """
    if max_chars <= 0:
        return [text]
    chunks: List[str] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + max_chars, text_len)
        chunks.append(text[start:end])
        if end == text_len:
            break
        start = max(0, end - overlap)
    return chunks


def normalize_whitespace(lines: Iterable[str]) -> List[str]:
    """Trim and collapse internal whitespace for each line."""
    return [" ".join(line.strip().split()) for line in lines]



"""Utilities for splitting text into semantic chunks."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ChunkMetadata:
    text: str
    index: int
    start_char: int
    end_char: int
    token_count: int

    def as_dict(self) -> Dict[str, int | str]:
        return {
            "text": self.text,
            "index": self.index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "token_count": self.token_count,
        }


class TextChunker:
    """Break text into overlapping chunks while keeping sentence boundaries."""

    sentence_pattern = re.compile(r"[^.!?]+(?:[.!?]+|$)", re.MULTILINE | re.DOTALL)

    def chunk(
        self, text: str, chunk_size: int = 512, overlap: int = 50
    ) -> List[Dict[str, int | str]]:
        if not text:
            return []

        sentences = self._split_into_sentences(text)

        chunks: List[ChunkMetadata] = []
        current: List[Dict[str, int | str]] = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = sentence["tokens"]

            if current and current_tokens + sent_tokens > chunk_size:
                chunks.append(self._finalize_chunk(current, len(chunks)))
                current, current_tokens = self._apply_overlap(current, overlap)

            current.append(sentence)
            current_tokens += sent_tokens

        if current:
            chunks.append(self._finalize_chunk(current, len(chunks)))

        return [chunk.as_dict() for chunk in chunks]

    def _split_into_sentences(self, text: str) -> List[Dict[str, int | str]]:
        sentences: List[Dict[str, int | str]] = []
        for match in self.sentence_pattern.finditer(text):
            sentence_text = match.group().strip()
            if not sentence_text:
                continue

            start = match.start()
            end = match.end()
            tokens = self._estimate_tokens(sentence_text)
            sentences.append(
                {
                    "text": sentence_text,
                    "start": start,
                    "end": end,
                    "tokens": tokens,
                }
            )

        if not sentences:
            sentences.append(
                {
                    "text": text.strip(),
                    "start": 0,
                    "end": len(text),
                    "tokens": self._estimate_tokens(text),
                }
            )
        return sentences

    def _apply_overlap(
        self, sentences: List[Dict[str, int | str]], overlap: int
    ) -> tuple[List[Dict[str, int | str]], int]:
        overlap_sentences: List[Dict[str, int | str]] = []
        tokens = 0

        for sentence in reversed(sentences):
            overlap_sentences.insert(0, sentence)
            tokens += sentence["tokens"]
            if tokens >= overlap:
                break

        return overlap_sentences, tokens

    def _finalize_chunk(
        self, sentences: List[Dict[str, int | str]], index: int
    ) -> ChunkMetadata:
        text = " ".join(sentence["text"] for sentence in sentences).strip()
        start_char = sentences[0]["start"]
        end_char = sentences[-1]["end"]
        token_count = sum(sentence["tokens"] for sentence in sentences)
        return ChunkMetadata(text, index, start_char, end_char, token_count)

    def _estimate_tokens(self, text: str) -> int:
        words = re.findall(r"\w+", text)
        approx = len(words) * 1.3
        return max(1, int(math.ceil(approx)))


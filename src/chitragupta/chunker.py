"""
Text chunking strategies for the RAG pipeline.

Provides multiple approaches to splitting document text into smaller
pieces suitable for embedding and retrieval.
"""

from __future__ import annotations

import re
from typing import List


class ChunkerBase:
    """Abstract base for chunking strategies."""

    def chunk(self, text: str) -> List[str]:
        """Split *text* into a list of chunk strings."""
        raise NotImplementedError


class FixedSizeChunker(ChunkerBase):
    """Split text into fixed-size character windows with optional overlap.

    Args:
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of overlapping characters between consecutive chunks.
    """

    def __init__(self, chunk_size: int = 200, overlap: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive, got {}".format(chunk_size))
        if overlap < 0:
            raise ValueError("overlap must be non-negative, got {}".format(overlap))
        if overlap >= chunk_size:
            raise ValueError(
                "overlap ({}) must be less than chunk_size ({})".format(overlap, chunk_size)
            )
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[str]:
        """Split *text* into overlapping fixed-size windows.

        Returns:
            List of chunk strings. Empty list if text is empty.
        """
        if not text:
            return []

        chunks: List[str] = []
        step = self.chunk_size - self.overlap
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            segment = text[start:end].strip()
            if segment:
                chunks.append(segment)
            start += step
        return chunks


class SentenceChunker(ChunkerBase):
    """Split text on sentence boundaries and group into chunks.

    Sentences are detected via a simple regex that splits on period,
    exclamation mark, or question mark followed by whitespace.

    Args:
        max_sentences: Maximum number of sentences per chunk.
    """

    _SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')

    def __init__(self, max_sentences: int = 5) -> None:
        if max_sentences <= 0:
            raise ValueError("max_sentences must be positive, got {}".format(max_sentences))
        self.max_sentences = max_sentences

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into individual sentences."""
        raw = self._SENTENCE_RE.split(text.strip())
        return [s.strip() for s in raw if s.strip()]

    def chunk(self, text: str) -> List[str]:
        """Group sentences into chunks of at most *max_sentences* each.

        Returns:
            List of chunk strings.
        """
        if not text:
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks: List[str] = []
        for i in range(0, len(sentences), self.max_sentences):
            group = sentences[i : i + self.max_sentences]
            joined = " ".join(group)
            if joined:
                chunks.append(joined)
        return chunks


class ParagraphChunker(ChunkerBase):
    """Split text on paragraph boundaries (double newlines).

    Consecutive blank lines are collapsed. Each non-empty paragraph
    becomes one chunk.

    Args:
        min_length: Minimum character length for a paragraph to be
            included as its own chunk. Shorter paragraphs are merged
            with the next one.
    """

    _PARA_RE = re.compile(r'\n\s*\n')

    def __init__(self, min_length: int = 20) -> None:
        if min_length < 0:
            raise ValueError("min_length must be non-negative, got {}".format(min_length))
        self.min_length = min_length

    def chunk(self, text: str) -> List[str]:
        """Split *text* on double-newline boundaries.

        Short paragraphs (below *min_length*) are merged with the
        following paragraph to avoid tiny chunks.

        Returns:
            List of paragraph chunk strings.
        """
        if not text:
            return []

        raw_paras = self._PARA_RE.split(text.strip())
        raw_paras = [p.strip() for p in raw_paras if p.strip()]

        if not raw_paras:
            return []

        # Merge short paragraphs with the next paragraph
        merged: List[str] = []
        buffer = ""
        for para in raw_paras:
            if buffer:
                buffer = buffer + " " + para
            else:
                buffer = para

            if len(buffer) >= self.min_length:
                merged.append(buffer)
                buffer = ""

        # Flush remaining buffer
        if buffer:
            if merged:
                merged[-1] = merged[-1] + " " + buffer
            else:
                merged.append(buffer)

        return merged

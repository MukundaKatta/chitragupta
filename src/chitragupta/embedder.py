"""
Simulated embedding generation for the RAG pipeline.

Uses deterministic hash-based pseudo-embeddings so the pipeline can be
tested and demonstrated without any ML library dependencies.
"""

from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Optional


class EmbeddingSimulator:
    """Generate deterministic pseudo-embedding vectors from text.

    The embedding is produced by hashing the text with SHA-512 and
    mapping byte values to the range [-1, 1]. The result is then
    L2-normalised so that cosine similarity behaves predictably.

    Args:
        dim: Dimensionality of the embedding vectors.
        seed: A string seed mixed into the hash for reproducibility.
    """

    def __init__(self, dim: int = 64, seed: str = "chitragupta") -> None:
        if dim <= 0:
            raise ValueError("dim must be positive, got {}".format(dim))
        self.dim = dim
        self.seed = seed

    def _hash_text(self, text: str) -> bytes:
        """Produce a deterministic byte sequence from *text*."""
        combined = self.seed + "|" + text.strip().lower()
        # Chain hashes to produce enough bytes for any dim
        result = b""
        current = combined.encode("utf-8")
        while len(result) < self.dim * 4:
            h = hashlib.sha512(current)
            result += h.digest()
            current = h.digest()
        return result

    @staticmethod
    def _normalise(vec: List[float]) -> List[float]:
        """L2-normalise a vector."""
        magnitude = math.sqrt(sum(v * v for v in vec))
        if magnitude == 0.0:
            return vec
        return [v / magnitude for v in vec]

    def embed(self, text: str) -> List[float]:
        """Compute a pseudo-embedding for *text*.

        Args:
            text: Input string.

        Returns:
            L2-normalised float vector of length *dim*.
        """
        raw = self._hash_text(text)
        # Map each byte to [-1, 1]
        vec = [(b / 127.5) - 1.0 for b in raw[: self.dim]]
        return self._normalise(vec)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for a batch of texts."""
        return [self.embed(t) for t in texts]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors.

    Both vectors are assumed to be the same length. Returns a value
    in [-1, 1].

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity score.

    Raises:
        ValueError: If vectors differ in length or are empty.
    """
    if len(a) != len(b):
        raise ValueError(
            "Vectors must have the same length ({} vs {})".format(len(a), len(b))
        )
    if not a:
        raise ValueError("Vectors must not be empty")

    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return dot / (mag_a * mag_b)


class EmbeddingCache:
    """Cache embeddings by text content to avoid redundant computation.

    Uses a simple dictionary keyed by the stripped, lowercased text.
    """

    def __init__(self) -> None:
        self._store: Dict[str, List[float]] = {}

    @staticmethod
    def _key(text: str) -> str:
        return text.strip().lower()

    def get(self, text: str) -> Optional[List[float]]:
        """Retrieve a cached embedding, or None if not found."""
        return self._store.get(self._key(text))

    def put(self, text: str, embedding: List[float]) -> None:
        """Store an embedding in the cache."""
        self._store[self._key(text)] = embedding

    def clear(self) -> None:
        """Remove all cached entries."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, text: str) -> bool:
        return self._key(text) in self._store

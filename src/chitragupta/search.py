"""
Semantic search over embedded chunks.

Provides brute-force similarity search with metadata filtering
and ranked result objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from chitragupta.embedder import EmbeddingSimulator, cosine_similarity


@dataclass
class SearchResult:
    """A single search result.

    Attributes:
        chunk_id: ID of the matching chunk.
        text: Text content of the matching chunk.
        score: Cosine similarity score (higher is more relevant).
        doc_id: ID of the parent document.
        position: Position of the chunk within its document.
        doc_metadata: Metadata inherited from the parent document.
    """

    chunk_id: str
    text: str
    score: float
    doc_id: str = ""
    position: int = 0
    doc_metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:60] + "..." if len(self.text) > 60 else self.text
        return "SearchResult(score={:.4f}, text='{}')".format(self.score, preview)


class SemanticSearch:
    """Brute-force semantic search over embedded chunks.

    Args:
        chunks: List of chunk-like objects with .id, .text, .doc_id,
            .position, .embedding, and .metadata attributes.
        doc_metadata: Mapping from document IDs to their metadata dicts.
        embedder: An EmbeddingSimulator used to embed queries.
    """

    def __init__(
        self,
        chunks: List[Any],
        doc_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        embedder: Optional[EmbeddingSimulator] = None,
    ) -> None:
        self._chunks = list(chunks)
        self._doc_metadata = doc_metadata or {}
        self._embedder = embedder or EmbeddingSimulator()

    @property
    def size(self) -> int:
        """Number of indexed chunks."""
        return len(self._chunks)

    def search(
        self,
        query: str,
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Find the top-k chunks most similar to *query*.

        Args:
            query: Natural-language query string.
            k: Maximum number of results to return.
            metadata_filter: If provided, only chunks whose doc_metadata
                contains all specified key-value pairs are considered.

        Returns:
            List of SearchResult objects sorted by descending score.
        """
        if k <= 0:
            return []

        query_embedding = self._embedder.embed(query)

        scored: List[SearchResult] = []
        for chunk in self._chunks:
            if not chunk.embedding:
                continue

            # Apply metadata filter
            doc_meta = self._doc_metadata.get(chunk.doc_id, {})
            if metadata_filter:
                if not self._matches_filter(doc_meta, metadata_filter):
                    continue

            score = cosine_similarity(query_embedding, chunk.embedding)
            result = SearchResult(
                chunk_id=chunk.id,
                text=chunk.text,
                score=score,
                doc_id=chunk.doc_id,
                position=chunk.position,
                doc_metadata=doc_meta,
            )
            scored.append(result)

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]

    @staticmethod
    def _matches_filter(
        metadata: Dict[str, Any], filters: Dict[str, Any]
    ) -> bool:
        """Check whether *metadata* contains all key-value pairs in *filters*."""
        for key, value in filters.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True

"""
Core RAG pipeline components.

Provides the end-to-end retrieval-augmented generation pipeline including
document and chunk data models, ingestion, chunking, embedding, indexing,
and search stages.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from chitragupta.chunker import (
    FixedSizeChunker,
    ParagraphChunker,
    SentenceChunker,
)
from chitragupta.embedder import EmbeddingSimulator, EmbeddingCache
from chitragupta.search import SemanticSearch, SearchResult


@dataclass
class Document:
    """Represents an ingested document with content and metadata.

    Attributes:
        content: The raw text content of the document.
        metadata: Arbitrary key-value metadata (author, date, etc.).
        source: Origin of the document (file path, URL, etc.).
        id: Unique identifier; auto-generated if not provided.
    """

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def __post_init__(self) -> None:
        if not self.content:
            raise ValueError("Document content must not be empty")

    @property
    def content_hash(self) -> str:
        """SHA-256 hash of the document content for deduplication."""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    @property
    def word_count(self) -> int:
        """Number of whitespace-delimited tokens in the content."""
        return len(self.content.split())

    def __len__(self) -> int:
        return len(self.content)


@dataclass
class Chunk:
    """A chunk of text extracted from a document.

    Attributes:
        text: The chunk text content.
        doc_id: ID of the parent document.
        position: Ordinal position within the parent document.
        embedding: Computed embedding vector (list of floats).
        id: Unique identifier; auto-generated if not provided.
        metadata: Inherited or chunk-specific metadata.
    """

    text: str
    doc_id: str = ""
    position: int = 0
    embedding: List[float] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("Chunk text must not be empty")

    @property
    def has_embedding(self) -> bool:
        """Whether an embedding vector has been computed."""
        return len(self.embedding) > 0

    @property
    def char_count(self) -> int:
        """Number of characters in the chunk text."""
        return len(self.text)

    def __len__(self) -> int:
        return len(self.text)


class PipelineError(Exception):
    """Raised when a pipeline operation fails."""
    pass


class RAGPipeline:
    """End-to-end retrieval-augmented generation pipeline.

    Orchestrates the full flow: ingest documents, chunk them, compute
    embeddings, build an index, and answer queries via semantic search.

    Attributes:
        documents: All ingested documents.
        chunks: All generated chunks.
        chunk_strategy: Name of the active chunking strategy.
        embedding_dim: Dimensionality of pseudo-embeddings.
    """

    VALID_STRATEGIES = ("fixed", "sentence", "paragraph")

    def __init__(
        self,
        chunk_strategy: str = "fixed",
        chunk_size: int = 200,
        chunk_overlap: int = 50,
        embedding_dim: int = 64,
    ) -> None:
        if chunk_strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                "Invalid chunk_strategy '{}'. Choose from: {}".format(
                    chunk_strategy, ", ".join(self.VALID_STRATEGIES)
                )
            )
        self.chunk_strategy = chunk_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_dim = embedding_dim

        self.documents: List[Document] = []
        self.chunks: List[Chunk] = []
        self._doc_hashes: set = set()

        self._embedder = EmbeddingSimulator(dim=embedding_dim)
        self._cache = EmbeddingCache()
        self._search_engine: Optional[SemanticSearch] = None
        self._indexed = False

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, documents: List[Document]) -> int:
        """Ingest a batch of documents, deduplicating by content hash.

        Args:
            documents: Documents to add to the pipeline.

        Returns:
            Number of new (non-duplicate) documents ingested.
        """
        added = 0
        for doc in documents:
            h = doc.content_hash
            if h not in self._doc_hashes:
                self._doc_hashes.add(h)
                self.documents.append(doc)
                added += 1
        self._indexed = False
        return added

    def ingest_text(self, text: str, metadata: Optional[Dict[str, Any]] = None,
                    source: str = "") -> Document:
        """Convenience: ingest a single raw text string.

        Returns:
            The created Document object.
        """
        doc = Document(content=text, metadata=metadata or {}, source=source)
        self.ingest([doc])
        return doc

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def chunk(self, strategy: Optional[str] = None) -> List[Chunk]:
        """Chunk all ingested documents using the configured strategy.

        Args:
            strategy: Override the default chunking strategy for this call.

        Returns:
            List of Chunk objects generated from all documents.
        """
        strat = strategy or self.chunk_strategy
        if strat not in self.VALID_STRATEGIES:
            raise PipelineError("Unknown chunking strategy: {}".format(strat))

        if strat == "fixed":
            chunker = FixedSizeChunker(
                chunk_size=self.chunk_size, overlap=self.chunk_overlap
            )
        elif strat == "sentence":
            chunker = SentenceChunker(max_sentences=5)
        else:
            chunker = ParagraphChunker()

        self.chunks = []
        for doc in self.documents:
            raw_chunks = chunker.chunk(doc.content)
            for i, text in enumerate(raw_chunks):
                chunk_obj = Chunk(
                    text=text,
                    doc_id=doc.id,
                    position=i,
                    metadata=dict(doc.metadata),
                )
                self.chunks.append(chunk_obj)

        self._indexed = False
        return self.chunks

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(self) -> int:
        """Compute embeddings for all chunks.

        Returns:
            Number of chunks that received new embeddings.
        """
        if not self.chunks:
            raise PipelineError("No chunks to embed. Call chunk() first.")

        count = 0
        for c in self.chunks:
            cached = self._cache.get(c.text)
            if cached is not None:
                c.embedding = cached
            else:
                emb = self._embedder.embed(c.text)
                c.embedding = emb
                self._cache.put(c.text, emb)
                count += 1
        self._indexed = False
        return count

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self) -> None:
        """Build the search index from embedded chunks."""
        embedded_chunks = [c for c in self.chunks if c.has_embedding]
        if not embedded_chunks:
            raise PipelineError(
                "No embedded chunks available. Call embed() first."
            )

        doc_map = {d.id: d.metadata for d in self.documents}
        self._search_engine = SemanticSearch(
            chunks=embedded_chunks,
            doc_metadata=doc_map,
            embedder=self._embedder,
        )
        self._indexed = True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 5,
               metadata_filter: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search for chunks most relevant to a query.

        Args:
            query: Natural language query string.
            k: Number of top results to return.
            metadata_filter: Optional key-value pairs that results must match.

        Returns:
            Ranked list of SearchResult objects.
        """
        if not self._indexed or self._search_engine is None:
            raise PipelineError("Index not built. Call index() first.")
        return self._search_engine.search(query, k=k, metadata_filter=metadata_filter)

    # ------------------------------------------------------------------
    # Pipeline convenience
    # ------------------------------------------------------------------

    def run(self, documents: List[Document], query: str, k: int = 5) -> List[SearchResult]:
        """Execute the full pipeline: ingest -> chunk -> embed -> index -> search.

        Args:
            documents: Documents to process.
            query: Query to search for after indexing.
            k: Number of results.

        Returns:
            Ranked list of SearchResult objects.
        """
        self.ingest(documents)
        self.chunk()
        self.embed()
        self.index()
        return self.search(query, k=k)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return pipeline statistics.

        Returns:
            Dictionary with counts and configuration info.
        """
        embedded_count = sum(1 for c in self.chunks if c.has_embedding)
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "embedded_chunks": embedded_count,
            "indexed": self._indexed,
            "chunk_strategy": self.chunk_strategy,
            "chunk_size": self.chunk_size,
            "embedding_dim": self.embedding_dim,
            "cache_size": len(self._cache),
        }

    def clear(self) -> None:
        """Reset the pipeline, removing all documents and chunks."""
        self.documents.clear()
        self.chunks.clear()
        self._doc_hashes.clear()
        self._cache.clear()
        self._search_engine = None
        self._indexed = False

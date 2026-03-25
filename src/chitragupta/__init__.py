"""
Chitragupta - Document Indexing and RAG Pipeline.

Named after the Hindu divine record keeper, Chitragupta provides
document ingestion, chunking, embedding simulation, and semantic search.
"""

__version__ = "0.1.0"

from chitragupta.core import RAGPipeline, Document, Chunk
from chitragupta.chunker import FixedSizeChunker, SentenceChunker, ParagraphChunker
from chitragupta.embedder import EmbeddingSimulator, cosine_similarity, EmbeddingCache
from chitragupta.search import SemanticSearch, SearchResult

__all__ = [
    "RAGPipeline",
    "Document",
    "Chunk",
    "FixedSizeChunker",
    "SentenceChunker",
    "ParagraphChunker",
    "EmbeddingSimulator",
    "cosine_similarity",
    "EmbeddingCache",
    "SemanticSearch",
    "SearchResult",
]

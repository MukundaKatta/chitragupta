"""
Configuration management for the Chitragupta RAG pipeline.

Loads settings from environment variables and .env files, providing
sensible defaults for all options.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class PipelineConfig:
    """Configuration for the RAG pipeline.

    All values can be overridden via environment variables prefixed
    with ``CHITRAGUPTA_``.

    Attributes:
        chunk_strategy: Chunking method (fixed, sentence, paragraph).
        chunk_size: Character count per chunk (fixed strategy).
        chunk_overlap: Overlap characters between chunks.
        embedding_dim: Dimensionality of pseudo-embeddings.
        search_top_k: Default number of search results.
        verbose: Enable verbose logging output.
    """

    chunk_strategy: str = "fixed"
    chunk_size: int = 200
    chunk_overlap: int = 50
    embedding_dim: int = 64
    search_top_k: int = 5
    verbose: bool = False

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Create a config populated from environment variables.

        Environment variable names follow the pattern
        ``CHITRAGUPTA_<FIELD>`` in upper-case.
        """
        return cls(
            chunk_strategy=os.environ.get("CHITRAGUPTA_CHUNK_STRATEGY", "fixed"),
            chunk_size=int(os.environ.get("CHITRAGUPTA_CHUNK_SIZE", "200")),
            chunk_overlap=int(os.environ.get("CHITRAGUPTA_CHUNK_OVERLAP", "50")),
            embedding_dim=int(os.environ.get("CHITRAGUPTA_EMBEDDING_DIM", "64")),
            search_top_k=int(os.environ.get("CHITRAGUPTA_SEARCH_TOP_K", "5")),
            verbose=os.environ.get("CHITRAGUPTA_VERBOSE", "").lower() in ("1", "true", "yes"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise configuration to a plain dictionary."""
        return {
            "chunk_strategy": self.chunk_strategy,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_dim": self.embedding_dim,
            "search_top_k": self.search_top_k,
            "verbose": self.verbose,
        }

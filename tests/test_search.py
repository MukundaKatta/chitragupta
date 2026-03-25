"""Tests for chitragupta.search — SemanticSearch and SearchResult."""

import pytest

from chitragupta.core import Chunk
from chitragupta.embedder import EmbeddingSimulator
from chitragupta.search import SearchResult, SemanticSearch


def _make_embedded_chunks(embedder):
    """Create a small set of embedded chunks for testing."""
    texts = [
        "The sacred Ganges flows through India",
        "Python is a popular programming language",
        "Machine learning uses neural networks",
        "The Himalayas are the tallest mountains",
        "Data science requires statistical knowledge",
    ]
    chunks = []
    for i, text in enumerate(texts):
        c = Chunk(
            text=text,
            doc_id="doc_{}".format(i % 2),
            position=i,
            embedding=embedder.embed(text),
        )
        chunks.append(c)
    return chunks


class TestSemanticSearch:
    def test_basic_search(self):
        emb = EmbeddingSimulator(dim=32)
        chunks = _make_embedded_chunks(emb)
        engine = SemanticSearch(chunks=chunks, embedder=emb)
        results = engine.search("river in India", k=2)
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)

    def test_results_sorted_by_score(self):
        emb = EmbeddingSimulator(dim=32)
        chunks = _make_embedded_chunks(emb)
        engine = SemanticSearch(chunks=chunks, embedder=emb)
        results = engine.search("programming", k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_k_limits_results(self):
        emb = EmbeddingSimulator(dim=32)
        chunks = _make_embedded_chunks(emb)
        engine = SemanticSearch(chunks=chunks, embedder=emb)
        results = engine.search("anything", k=2)
        assert len(results) <= 2

    def test_k_zero_returns_empty(self):
        emb = EmbeddingSimulator(dim=32)
        chunks = _make_embedded_chunks(emb)
        engine = SemanticSearch(chunks=chunks, embedder=emb)
        assert engine.search("query", k=0) == []

    def test_metadata_filter(self):
        emb = EmbeddingSimulator(dim=32)
        chunks = _make_embedded_chunks(emb)
        doc_meta = {
            "doc_0": {"category": "geography"},
            "doc_1": {"category": "tech"},
        }
        engine = SemanticSearch(chunks=chunks, doc_metadata=doc_meta, embedder=emb)
        results = engine.search("river", k=10, metadata_filter={"category": "geography"})
        for r in results:
            assert r.doc_metadata.get("category") == "geography"

    def test_size_property(self):
        emb = EmbeddingSimulator(dim=32)
        chunks = _make_embedded_chunks(emb)
        engine = SemanticSearch(chunks=chunks, embedder=emb)
        assert engine.size == len(chunks)

    def test_search_result_repr(self):
        r = SearchResult(chunk_id="c1", text="short", score=0.95)
        assert "0.9500" in repr(r)
        assert "short" in repr(r)

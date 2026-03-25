"""Tests for chitragupta.core — Document, Chunk, and RAGPipeline."""

import pytest

from chitragupta.core import Chunk, Document, PipelineError, RAGPipeline


# ------------------------------------------------------------------
# Document tests
# ------------------------------------------------------------------

class TestDocument:
    def test_create_document(self):
        doc = Document(content="Hello world", source="test.txt")
        assert doc.content == "Hello world"
        assert doc.source == "test.txt"
        assert len(doc.id) == 12

    def test_document_requires_content(self):
        with pytest.raises(ValueError, match="content must not be empty"):
            Document(content="")

    def test_document_content_hash_deterministic(self):
        d1 = Document(content="same text")
        d2 = Document(content="same text")
        assert d1.content_hash == d2.content_hash

    def test_document_content_hash_differs(self):
        d1 = Document(content="text A")
        d2 = Document(content="text B")
        assert d1.content_hash != d2.content_hash

    def test_document_word_count(self):
        doc = Document(content="one two three four five")
        assert doc.word_count == 5

    def test_document_len(self):
        doc = Document(content="abcdef")
        assert len(doc) == 6

    def test_document_metadata(self):
        doc = Document(content="x", metadata={"author": "Chitragupta"})
        assert doc.metadata["author"] == "Chitragupta"


# ------------------------------------------------------------------
# Chunk tests
# ------------------------------------------------------------------

class TestChunk:
    def test_create_chunk(self):
        c = Chunk(text="sample", doc_id="d1", position=0)
        assert c.text == "sample"
        assert c.doc_id == "d1"

    def test_chunk_requires_text(self):
        with pytest.raises(ValueError, match="text must not be empty"):
            Chunk(text="")

    def test_chunk_has_embedding_false(self):
        c = Chunk(text="hello")
        assert c.has_embedding is False

    def test_chunk_has_embedding_true(self):
        c = Chunk(text="hello", embedding=[0.1, 0.2])
        assert c.has_embedding is True

    def test_chunk_char_count(self):
        c = Chunk(text="abcdef")
        assert c.char_count == 6


# ------------------------------------------------------------------
# RAGPipeline tests
# ------------------------------------------------------------------

class TestRAGPipeline:
    def _sample_docs(self):
        return [
            Document(
                content="The Ganges is a sacred river in India. "
                        "It flows through the plains of northern India. "
                        "Millions of people depend on the Ganges for water.",
                metadata={"topic": "geography"},
                source="ganges.txt",
            ),
            Document(
                content="Python is a versatile programming language. "
                        "It is widely used for data science and web development. "
                        "Python has a large ecosystem of libraries.",
                metadata={"topic": "technology"},
                source="python.txt",
            ),
        ]

    def test_invalid_strategy(self):
        with pytest.raises(ValueError, match="Invalid chunk_strategy"):
            RAGPipeline(chunk_strategy="unknown")

    def test_ingest_documents(self):
        pipe = RAGPipeline()
        docs = self._sample_docs()
        added = pipe.ingest(docs)
        assert added == 2
        assert len(pipe.documents) == 2

    def test_ingest_deduplication(self):
        pipe = RAGPipeline()
        docs = self._sample_docs()
        pipe.ingest(docs)
        added2 = pipe.ingest(docs)
        assert added2 == 0
        assert len(pipe.documents) == 2

    def test_ingest_text_convenience(self):
        pipe = RAGPipeline()
        doc = pipe.ingest_text("Some text content", source="inline")
        assert doc.source == "inline"
        assert len(pipe.documents) == 1

    def test_chunk_fixed(self):
        pipe = RAGPipeline(chunk_strategy="fixed", chunk_size=50, chunk_overlap=10)
        pipe.ingest(self._sample_docs())
        chunks = pipe.chunk()
        assert len(chunks) > 0
        for c in chunks:
            assert len(c.text) <= 50

    def test_chunk_sentence(self):
        pipe = RAGPipeline(chunk_strategy="sentence")
        pipe.ingest(self._sample_docs())
        chunks = pipe.chunk()
        assert len(chunks) > 0

    def test_chunk_paragraph(self):
        pipe = RAGPipeline(chunk_strategy="paragraph")
        doc = Document(content="First paragraph.\n\nSecond paragraph.\n\nThird paragraph.")
        pipe.ingest([doc])
        chunks = pipe.chunk()
        assert len(chunks) >= 1

    def test_embed_requires_chunks(self):
        pipe = RAGPipeline()
        with pytest.raises(PipelineError, match="No chunks"):
            pipe.embed()

    def test_embed_chunks(self):
        pipe = RAGPipeline(embedding_dim=32)
        pipe.ingest(self._sample_docs())
        pipe.chunk()
        count = pipe.embed()
        assert count > 0
        for c in pipe.chunks:
            assert c.has_embedding
            assert len(c.embedding) == 32

    def test_index_requires_embeddings(self):
        pipe = RAGPipeline()
        pipe.ingest(self._sample_docs())
        pipe.chunk()
        with pytest.raises(PipelineError, match="No embedded chunks"):
            pipe.index()

    def test_search_requires_index(self):
        pipe = RAGPipeline()
        with pytest.raises(PipelineError, match="Index not built"):
            pipe.search("query")

    def test_full_pipeline_run(self):
        pipe = RAGPipeline(chunk_size=80, chunk_overlap=20, embedding_dim=32)
        results = pipe.run(self._sample_docs(), "sacred river India")
        assert len(results) > 0
        assert results[0].score > -1.0

    def test_search_with_metadata_filter(self):
        pipe = RAGPipeline(chunk_size=80, chunk_overlap=20)
        pipe.ingest(self._sample_docs())
        pipe.chunk()
        pipe.embed()
        pipe.index()
        results = pipe.search("river", k=10, metadata_filter={"topic": "geography"})
        for r in results:
            assert r.doc_metadata.get("topic") == "geography"

    def test_stats(self):
        pipe = RAGPipeline()
        pipe.ingest(self._sample_docs())
        pipe.chunk()
        pipe.embed()
        s = pipe.stats()
        assert s["documents"] == 2
        assert s["chunks"] > 0
        assert s["embedded_chunks"] > 0

    def test_clear_pipeline(self):
        pipe = RAGPipeline()
        pipe.ingest(self._sample_docs())
        pipe.chunk()
        pipe.embed()
        pipe.clear()
        assert len(pipe.documents) == 0
        assert len(pipe.chunks) == 0

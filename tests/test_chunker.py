"""Tests for chitragupta.chunker — FixedSizeChunker, SentenceChunker, ParagraphChunker."""

import pytest

from chitragupta.chunker import FixedSizeChunker, ParagraphChunker, SentenceChunker


# ------------------------------------------------------------------
# FixedSizeChunker
# ------------------------------------------------------------------

class TestFixedSizeChunker:
    def test_basic_chunking(self):
        chunker = FixedSizeChunker(chunk_size=10, overlap=0)
        result = chunker.chunk("abcdefghijklmnopqrstuvwxyz")
        assert len(result) == 3
        assert result[0] == "abcdefghij"

    def test_overlap(self):
        chunker = FixedSizeChunker(chunk_size=10, overlap=5)
        result = chunker.chunk("abcdefghijklmnopqrstuvwxyz")
        # With overlap of 5, step is 5 chars
        assert len(result) >= 4
        # Second chunk should start 5 chars in
        assert result[1].startswith("fghij")

    def test_empty_text(self):
        chunker = FixedSizeChunker(chunk_size=10, overlap=0)
        assert chunker.chunk("") == []

    def test_text_shorter_than_chunk_size(self):
        chunker = FixedSizeChunker(chunk_size=100, overlap=0)
        result = chunker.chunk("short text")
        assert len(result) == 1
        assert result[0] == "short text"

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            FixedSizeChunker(chunk_size=0)

    def test_invalid_overlap(self):
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            FixedSizeChunker(chunk_size=10, overlap=-1)

    def test_overlap_exceeds_chunk_size(self):
        with pytest.raises(ValueError, match="overlap.*must be less than chunk_size"):
            FixedSizeChunker(chunk_size=10, overlap=10)

    def test_all_chunks_within_size(self):
        chunker = FixedSizeChunker(chunk_size=20, overlap=5)
        text = "The quick brown fox jumps over the lazy dog. " * 5
        for chunk in chunker.chunk(text):
            assert len(chunk) <= 20


# ------------------------------------------------------------------
# SentenceChunker
# ------------------------------------------------------------------

class TestSentenceChunker:
    def test_basic_sentence_split(self):
        chunker = SentenceChunker(max_sentences=1)
        text = "First sentence. Second sentence. Third sentence."
        result = chunker.chunk(text)
        assert len(result) == 3

    def test_grouped_sentences(self):
        chunker = SentenceChunker(max_sentences=2)
        text = "One. Two. Three. Four."
        result = chunker.chunk(text)
        assert len(result) == 2

    def test_empty_text(self):
        chunker = SentenceChunker()
        assert chunker.chunk("") == []

    def test_single_sentence(self):
        chunker = SentenceChunker(max_sentences=3)
        result = chunker.chunk("Just one sentence.")
        assert len(result) == 1

    def test_question_marks(self):
        chunker = SentenceChunker(max_sentences=1)
        text = "What is this? Is it a test? Yes it is."
        result = chunker.chunk(text)
        assert len(result) == 3

    def test_exclamation_marks(self):
        chunker = SentenceChunker(max_sentences=1)
        text = "Wow! Amazing! Incredible!"
        result = chunker.chunk(text)
        assert len(result) == 3

    def test_invalid_max_sentences(self):
        with pytest.raises(ValueError, match="max_sentences must be positive"):
            SentenceChunker(max_sentences=0)

    def test_preserves_text_content(self):
        chunker = SentenceChunker(max_sentences=5)
        text = "Hello world. Goodbye world."
        result = chunker.chunk(text)
        joined = " ".join(result)
        assert "Hello world." in joined
        assert "Goodbye world." in joined


# ------------------------------------------------------------------
# ParagraphChunker
# ------------------------------------------------------------------

class TestParagraphChunker:
    def test_basic_paragraph_split(self):
        chunker = ParagraphChunker(min_length=1)
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        result = chunker.chunk(text)
        assert len(result) == 3

    def test_merges_short_paragraphs(self):
        chunker = ParagraphChunker(min_length=30)
        text = "Hi.\n\nBye.\n\nThis is a longer paragraph that exceeds thirty chars."
        result = chunker.chunk(text)
        # Short paras merged together or into longer ones
        assert len(result) < 3

    def test_empty_text(self):
        chunker = ParagraphChunker()
        assert chunker.chunk("") == []

    def test_single_paragraph(self):
        chunker = ParagraphChunker(min_length=1)
        result = chunker.chunk("Just one paragraph with no breaks.")
        assert len(result) == 1

    def test_multiple_blank_lines(self):
        chunker = ParagraphChunker(min_length=1)
        text = "Para one.\n\n\n\nPara two."
        result = chunker.chunk(text)
        assert len(result) == 2

    def test_invalid_min_length(self):
        with pytest.raises(ValueError, match="min_length must be non-negative"):
            ParagraphChunker(min_length=-1)

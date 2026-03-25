"""Tests for chitragupta.embedder — EmbeddingSimulator, cosine_similarity, EmbeddingCache."""

import math

import pytest

from chitragupta.embedder import EmbeddingCache, EmbeddingSimulator, cosine_similarity


class TestEmbeddingSimulator:
    def test_embed_returns_correct_dim(self):
        emb = EmbeddingSimulator(dim=32)
        vec = emb.embed("hello")
        assert len(vec) == 32

    def test_embed_deterministic(self):
        emb = EmbeddingSimulator(dim=16)
        v1 = emb.embed("test text")
        v2 = emb.embed("test text")
        assert v1 == v2

    def test_embed_normalised(self):
        emb = EmbeddingSimulator(dim=64)
        vec = emb.embed("some document text")
        magnitude = math.sqrt(sum(v * v for v in vec))
        assert abs(magnitude - 1.0) < 1e-6

    def test_different_texts_different_embeddings(self):
        emb = EmbeddingSimulator(dim=16)
        v1 = emb.embed("apples")
        v2 = emb.embed("oranges")
        assert v1 != v2

    def test_embed_batch(self):
        emb = EmbeddingSimulator(dim=8)
        results = emb.embed_batch(["one", "two", "three"])
        assert len(results) == 3
        assert all(len(v) == 8 for v in results)

    def test_invalid_dim(self):
        with pytest.raises(ValueError, match="dim must be positive"):
            EmbeddingSimulator(dim=0)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        vec = [0.5, 0.3, 0.8]
        assert abs(cosine_similarity(vec, vec) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            cosine_similarity([1.0], [1.0, 2.0])

    def test_empty_vectors(self):
        with pytest.raises(ValueError, match="must not be empty"):
            cosine_similarity([], [])


class TestEmbeddingCache:
    def test_put_and_get(self):
        cache = EmbeddingCache()
        cache.put("hello", [0.1, 0.2])
        assert cache.get("hello") == [0.1, 0.2]

    def test_get_missing(self):
        cache = EmbeddingCache()
        assert cache.get("nope") is None

    def test_case_insensitive(self):
        cache = EmbeddingCache()
        cache.put("Hello World", [0.5])
        assert cache.get("hello world") == [0.5]

    def test_len_and_contains(self):
        cache = EmbeddingCache()
        cache.put("a", [1.0])
        cache.put("b", [2.0])
        assert len(cache) == 2
        assert "a" in cache
        assert "c" not in cache

    def test_clear(self):
        cache = EmbeddingCache()
        cache.put("x", [0.0])
        cache.clear()
        assert len(cache) == 0
        assert cache.get("x") is None

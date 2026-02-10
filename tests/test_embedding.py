"""Tests for embedding helpers."""

from __future__ import annotations

import pytest

from clio_pipeline.pipeline import EmbeddingExtractionError, embed_texts_in_batches


class _FakeEmbeddingClient:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), float(index), 1.0] for index, text in enumerate(texts)]


class _BrokenEmbeddingClient:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [[1.0, 2.0]] * (len(texts) - 1)


class TestEmbedTextsInBatches:
    def test_batches_return_2d_array(self):
        client = _FakeEmbeddingClient()
        embeddings = embed_texts_in_batches(
            ["alpha", "beta", "gamma", "delta"],
            client,
            batch_size=2,
        )
        assert embeddings.shape == (4, 3)

    def test_raises_for_empty_input(self):
        client = _FakeEmbeddingClient()
        with pytest.raises(EmbeddingExtractionError, match="empty text list"):
            embed_texts_in_batches([], client)

    def test_raises_for_mismatched_counts(self):
        client = _BrokenEmbeddingClient()
        with pytest.raises(EmbeddingExtractionError, match="Embedding count mismatch"):
            embed_texts_in_batches(["a", "b", "c"], client, batch_size=3)

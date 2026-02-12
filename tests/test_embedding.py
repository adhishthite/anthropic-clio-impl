"""Tests for embedding helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from clio_pipeline.pipeline import EmbeddingExtractionError, embed_texts_in_batches


class _FakeEmbeddingClient:
    def __init__(self):
        self.call_count = 0

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        return [[float(len(text)), float(index), 1.0] for index, text in enumerate(texts)]


class _BrokenEmbeddingClient:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [[1.0, 2.0]] * (len(texts) - 1)


class _FailOnBatchClient:
    """Succeeds for the first N batches, then raises on the next."""

    def __init__(self, fail_after_batches: int):
        self._fail_after = fail_after_batches
        self._batch_count = 0

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self._batch_count += 1
        if self._batch_count > self._fail_after:
            raise ConnectionError("Simulated network failure")
        return [[float(len(t)), 1.0, 2.0] for t in texts]


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

    def test_checkpoint_saves_and_resumes(self, tmp_path: Path):
        texts = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
        checkpoint = tmp_path / "embed.partial.npy"

        # First run: fail after 2 batches (4 texts embedded at batch_size=2)
        client_fail = _FailOnBatchClient(fail_after_batches=2)
        with pytest.raises(ConnectionError):
            embed_texts_in_batches(texts, client_fail, batch_size=2, checkpoint_path=checkpoint)
        assert checkpoint.exists()
        partial = np.load(checkpoint)
        assert partial.shape[0] == 4

        # Resume: should only embed the remaining 2 texts
        client_ok = _FakeEmbeddingClient()
        result = embed_texts_in_batches(texts, client_ok, batch_size=2, checkpoint_path=checkpoint)
        assert result.shape == (6, 3)
        assert client_ok.call_count == 1  # only 1 batch for remaining 2 texts
        assert not checkpoint.exists()  # cleaned up on success

    def test_checkpoint_no_op_without_path(self):
        client = _FakeEmbeddingClient()
        result = embed_texts_in_batches(["a", "b"], client, batch_size=2, checkpoint_path=None)
        assert result.shape == (2, 3)

    def test_checkpoint_cleaned_up_on_full_success(self, tmp_path: Path):
        checkpoint = tmp_path / "embed.partial.npy"
        client = _FakeEmbeddingClient()
        result = embed_texts_in_batches(
            ["a", "b", "c"], client, batch_size=2, checkpoint_path=checkpoint
        )
        assert result.shape == (3, 3)
        assert not checkpoint.exists()

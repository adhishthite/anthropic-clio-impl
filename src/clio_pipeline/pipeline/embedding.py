"""Embedding helpers for Phase 3 clustering."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from clio_pipeline.models import TextEmbeddingClient


class EmbeddingExtractionError(ValueError):
    """Raised when text embeddings are missing or malformed."""


def embed_texts_in_batches(
    texts: list[str],
    embedding_client: TextEmbeddingClient,
    *,
    batch_size: int = 32,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Embed text inputs in batches and return a 2D array."""

    if batch_size <= 0:
        raise EmbeddingExtractionError(f"batch_size must be positive, got {batch_size}.")

    if not texts:
        raise EmbeddingExtractionError("Cannot embed an empty text list.")

    vectors: list[list[float]] = []
    expected_dim: int | None = None

    for start_idx in range(0, len(texts), batch_size):
        batch = texts[start_idx : start_idx + batch_size]
        batch_vectors = embedding_client.embed_texts(batch)
        if len(batch_vectors) != len(batch):
            raise EmbeddingExtractionError(
                f"Embedding count mismatch for batch starting at {start_idx}: "
                f"{len(batch_vectors)} != {len(batch)}."
            )

        for vector in batch_vectors:
            if expected_dim is None:
                expected_dim = len(vector)
            elif len(vector) != expected_dim:
                raise EmbeddingExtractionError(
                    "Inconsistent embedding dimensions: "
                    f"expected {expected_dim}, got {len(vector)}."
                )
            vectors.append(vector)
        if progress_callback is not None:
            progress_callback(len(vectors), len(texts))

    return np.asarray(vectors, dtype=float)

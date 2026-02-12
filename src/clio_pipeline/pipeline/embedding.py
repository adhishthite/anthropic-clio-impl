"""Embedding helpers for Phase 3 clustering."""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

from clio_pipeline.models import TextEmbeddingClient

logger = logging.getLogger(__name__)


class EmbeddingExtractionError(ValueError):
    """Raised when text embeddings are missing or malformed."""


def _save_checkpoint(checkpoint_path: Path, arr: np.ndarray) -> None:
    """Atomically save embedding checkpoint via temp file + rename."""
    import os

    parent = checkpoint_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=parent, suffix=".npy")
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        np.save(tmp_path, arr)
        tmp_path.rename(checkpoint_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _load_checkpoint(checkpoint_path: Path) -> np.ndarray | None:
    """Load partial embeddings from checkpoint if it exists."""
    if not checkpoint_path.exists():
        return None
    try:
        arr = np.load(checkpoint_path)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return None
        return arr
    except Exception:
        logger.warning("Corrupt embedding checkpoint at %s, starting fresh.", checkpoint_path)
        return None


def embed_texts_in_batches(
    texts: list[str],
    embedding_client: TextEmbeddingClient,
    *,
    batch_size: int = 32,
    progress_callback: Callable[[int, int], None] | None = None,
    checkpoint_path: Path | None = None,
) -> np.ndarray:
    """Embed text inputs in batches and return a 2D array.

    If checkpoint_path is provided, partial results are saved after each batch
    and resumed on restart. The checkpoint file is removed on successful completion.
    """

    if batch_size <= 0:
        raise EmbeddingExtractionError(f"batch_size must be positive, got {batch_size}.")

    if not texts:
        raise EmbeddingExtractionError("Cannot embed an empty text list.")

    vectors: list[list[float]] = []
    expected_dim: int | None = None
    resumed_count = 0

    if checkpoint_path is not None:
        partial = _load_checkpoint(checkpoint_path)
        if partial is not None:
            resumed_count = partial.shape[0]
            if resumed_count >= len(texts):
                logger.info("Checkpoint covers all %d texts, skipping embedding.", len(texts))
                checkpoint_path.unlink(missing_ok=True)
                return partial[: len(texts)]
            expected_dim = partial.shape[1]
            vectors = [list(row) for row in partial]
            logger.info("Resumed %d/%d embeddings from checkpoint.", resumed_count, len(texts))
            if progress_callback is not None:
                progress_callback(resumed_count, len(texts))

    for start_idx in range(resumed_count, len(texts), batch_size):
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

        if checkpoint_path is not None:
            _save_checkpoint(checkpoint_path, np.asarray(vectors, dtype=float))

        if progress_callback is not None:
            progress_callback(len(vectors), len(texts))

    result = np.asarray(vectors, dtype=float)

    if checkpoint_path is not None:
        checkpoint_path.unlink(missing_ok=True)

    return result

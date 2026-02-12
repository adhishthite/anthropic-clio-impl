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

    n_texts = len(texts)
    result: np.ndarray | None = None
    filled = 0

    if checkpoint_path is not None:
        partial = _load_checkpoint(checkpoint_path)
        if partial is not None:
            if partial.shape[0] >= n_texts:
                logger.info("Checkpoint covers all %d texts, skipping embedding.", n_texts)
                checkpoint_path.unlink(missing_ok=True)
                return partial[:n_texts]
            result = np.empty((n_texts, partial.shape[1]), dtype=float)
            result[: partial.shape[0]] = partial
            filled = partial.shape[0]
            logger.info("Resumed %d/%d embeddings from checkpoint.", filled, n_texts)
            if progress_callback is not None:
                progress_callback(filled, n_texts)

    for start_idx in range(filled, n_texts, batch_size):
        batch = texts[start_idx : start_idx + batch_size]
        batch_vectors = embedding_client.embed_texts(batch)
        if len(batch_vectors) != len(batch):
            raise EmbeddingExtractionError(
                f"Embedding count mismatch for batch starting at {start_idx}: "
                f"{len(batch_vectors)} != {len(batch)}."
            )

        if result is None:
            dim = len(batch_vectors[0])
            result = np.empty((n_texts, dim), dtype=float)

        for i, vector in enumerate(batch_vectors):
            if len(vector) != result.shape[1]:
                raise EmbeddingExtractionError(
                    "Inconsistent embedding dimensions: "
                    f"expected {result.shape[1]}, got {len(vector)}."
                )
            result[filled + i] = vector
        filled += len(batch_vectors)

        if checkpoint_path is not None:
            _save_checkpoint(checkpoint_path, result[:filled])

        if progress_callback is not None:
            progress_callback(filled, n_texts)

    if result is None:
        raise EmbeddingExtractionError("No embeddings produced.")

    if checkpoint_path is not None:
        checkpoint_path.unlink(missing_ok=True)

    return result

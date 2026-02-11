"""Jina embeddings client wrapper."""

from __future__ import annotations

from typing import Protocol

import httpx
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential, wait_random


class TextEmbeddingClient(Protocol):
    """Protocol for text embedding clients."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""


class JinaEmbeddingClient:
    """Thin client around Jina's embeddings endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.jina.ai/v1/embeddings",
        timeout_seconds: float = 60.0,
        max_retries: int = 4,
        backoff_seconds: float = 1.0,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._http = httpx.Client(
            timeout=timeout_seconds,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    def close(self) -> None:
        """Close the underlying HTTP client."""

        self._http.close()

    def _is_retryable_jina_error(self, exc: BaseException) -> bool:
        """Return whether a Jina request exception should trigger retry/backoff."""

        return isinstance(exc, (httpx.TimeoutException, httpx.RequestError, httpx.HTTPStatusError))

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using Jina."""

        if not texts:
            return []

        response: httpx.Response | None = None
        max_attempts = max(1, self._max_retries)
        wait_strategy = wait_exponential(
            multiplier=self._backoff_seconds,
            min=self._backoff_seconds,
            max=max(self._backoff_seconds, self._backoff_seconds * 8),
        ) + wait_random(0.0, 0.25)
        retryer = Retrying(
            retry=retry_if_exception(self._is_retryable_jina_error),
            wait=wait_strategy,
            stop=stop_after_attempt(max_attempts),
            reraise=True,
        )
        for attempt in retryer:
            with attempt:
                response = self._http.post(
                    self._base_url,
                    json={
                        "model": self._model,
                        "input": texts,
                    },
                )
                response.raise_for_status()

        if response is None:
            raise ValueError("Jina embeddings response missing after retries.")

        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected embeddings response type: {type(payload).__name__}")

        data = payload.get("data")
        if not isinstance(data, list):
            raise ValueError("Embeddings response missing list field 'data'.")

        embeddings_by_index: dict[int, list[float]] = {}
        for item in data:
            if not isinstance(item, dict):
                raise ValueError("Embeddings response 'data' contains non-object entries.")

            index = item.get("index")
            embedding = item.get("embedding")
            if not isinstance(index, int):
                raise ValueError("Embedding item missing integer 'index'.")
            if not isinstance(embedding, list):
                raise ValueError("Embedding item missing list 'embedding'.")

            vector = [float(value) for value in embedding]
            embeddings_by_index[index] = vector

        if len(embeddings_by_index) != len(texts):
            raise ValueError(
                "Embeddings response count does not match input count: "
                f"{len(embeddings_by_index)} != {len(texts)}."
            )

        return [embeddings_by_index[idx] for idx in range(len(texts))]

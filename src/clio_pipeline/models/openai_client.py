"""OpenAI client wrapper used by the pipeline."""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from typing import Protocol

from openai import APIError, APITimeoutError, BadRequestError, OpenAI, RateLimitError
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential, wait_random

from clio_pipeline.observability import maybe_wrap_openai_client


class LLMJsonClient(Protocol):
    """Protocol for clients that return structured JSON."""

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str | None = None,
        json_schema: dict | None = None,
        strict_schema: bool = True,
    ) -> dict:
        """Generate a JSON object for the given prompts."""


class OpenAIJsonClient:
    """JSON-focused wrapper around OpenAI chat completions."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_retries: int = 4,
        backoff_seconds: float = 1.0,
    ) -> None:
        base_client = OpenAI(api_key=api_key, base_url=base_url or None)
        self._client, self._langsmith_wrapped = maybe_wrap_openai_client(base_client)
        self._model = model
        self._temperature = temperature
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._metrics_lock = threading.Lock()
        self._request_count = 0
        self._retry_count = 0
        self._schema_fallback_count = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0

    def _supports_schema_fallback(self, exc: BadRequestError) -> bool:
        """Return True when the error suggests schema response format is unsupported."""

        message = str(exc).lower()
        fallback_tokens: Sequence[str] = (
            "json_schema",
            "response_format",
            "unsupported",
            "not supported",
            "invalid schema",
        )
        return any(token in message for token in fallback_tokens)

    def _is_retryable_openai_error(self, exc: BaseException) -> bool:
        """Return whether an OpenAI exception should trigger retry/backoff."""

        if isinstance(exc, (RateLimitError, APITimeoutError)):
            return True
        if isinstance(exc, BadRequestError):
            return False
        return isinstance(exc, APIError)

    def _create_completion_with_retry(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_format: dict,
    ):
        """Create chat completion with retry/backoff for transient errors."""

        response = None
        attempt_count = 0
        max_attempts = max(1, self._max_retries)
        wait_strategy = wait_exponential(
            multiplier=self._backoff_seconds,
            min=self._backoff_seconds,
            max=max(self._backoff_seconds, self._backoff_seconds * 8),
        ) + wait_random(0.0, 0.25)
        retryer = Retrying(
            retry=retry_if_exception(self._is_retryable_openai_error),
            wait=wait_strategy,
            stop=stop_after_attempt(max_attempts),
            reraise=True,
        )

        for attempt in retryer:
            with attempt:
                attempt_count += 1
                response = self._client.chat.completions.create(
                    model=self._model,
                    temperature=self._temperature,
                    response_format=response_format,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )

        if response is None:
            raise ValueError("OpenAI response missing after retries.")

        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        with self._metrics_lock:
            self._request_count += 1
            self._retry_count += max(0, attempt_count - 1)
            self._prompt_tokens += prompt_tokens
            self._completion_tokens += completion_tokens
            self._total_tokens += total_tokens
        return response

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str | None = None,
        json_schema: dict | None = None,
        strict_schema: bool = True,
    ) -> dict:
        """Call the OpenAI API and parse a JSON object from the response."""

        schema_response_format = None
        if json_schema is not None:
            schema_response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name or "structured_output",
                    "schema": json_schema,
                    "strict": bool(strict_schema),
                },
            }

        if schema_response_format is not None:
            try:
                response = self._create_completion_with_retry(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_format=schema_response_format,
                )
            except BadRequestError as exc:
                if not self._supports_schema_fallback(exc):
                    raise
                with self._metrics_lock:
                    self._schema_fallback_count += 1
                response = self._create_completion_with_retry(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_format={"type": "json_object"},
                )
        else:
            response = self._create_completion_with_retry(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format={"type": "json_object"},
            )

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Model returned empty content for JSON response.")

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Model response was not valid JSON: {content}") from exc

        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object, got {type(payload).__name__}.")

        return payload

    def metrics_snapshot(self) -> dict:
        """Return cumulative request/usage metrics for this client instance."""

        with self._metrics_lock:
            return {
                "request_count": self._request_count,
                "retry_count": self._retry_count,
                "schema_fallback_count": self._schema_fallback_count,
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "total_tokens": self._total_tokens,
                "model": self._model,
            }

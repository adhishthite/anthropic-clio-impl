"""OpenAI client wrapper used by the pipeline."""

from __future__ import annotations

import json
import random
import time
from collections.abc import Sequence
from typing import Protocol

from openai import APIError, APITimeoutError, BadRequestError, OpenAI, RateLimitError

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

    def _create_completion_with_retry(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_format: dict,
    ):
        """Create chat completion with retry/backoff for transient errors."""

        last_error: Exception | None = None
        response = None
        for attempt in range(self._max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    temperature=self._temperature,
                    response_format=response_format,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                break
            except (RateLimitError, APITimeoutError, APIError) as exc:
                last_error = exc
                if attempt >= self._max_retries - 1:
                    raise
                sleep_seconds = (self._backoff_seconds * (2**attempt)) + random.uniform(0.0, 0.25)
                time.sleep(sleep_seconds)

        if response is None:
            raise ValueError(f"OpenAI response missing after retries: {last_error}")
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

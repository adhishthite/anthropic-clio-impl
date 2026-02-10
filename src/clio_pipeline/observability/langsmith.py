"""LangSmith tracing helpers for local pipeline runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_LANGSMITH_ENV_KEYS = {
    "LANGSMITH_TRACING",
    "LANGSMITH_ENDPOINT",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_ENDPOINT",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_PROJECT",
}


def _parse_dotenv(path: Path) -> dict[str, str]:
    """Parse key-value pairs from dotenv file."""

    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        value = raw_value.strip().strip('"').strip("'")
        values[key] = value
    return values


def _is_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def load_langsmith_env_from_dotenv(path: str | Path = ".env") -> None:
    """Load LangSmith-related keys from dotenv into process env when missing."""

    dotenv_path = Path(path)
    dotenv_values = _parse_dotenv(dotenv_path)
    for key, value in dotenv_values.items():
        if key in _LANGSMITH_ENV_KEYS and not os.getenv(key):
            os.environ[key] = value


def _effective_tracing_flag() -> str:
    return os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2") or ""


def get_langsmith_status(path: str | Path = ".env") -> dict[str, Any]:
    """Return effective LangSmith tracing status after dotenv hydration."""

    load_langsmith_env_from_dotenv(path)
    tracing_value = _effective_tracing_flag()
    enabled = _is_truthy(tracing_value)
    endpoint = os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT") or ""
    project = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or ""
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY") or ""
    return {
        "enabled": enabled,
        "tracing_value": tracing_value,
        "endpoint": endpoint,
        "project": project,
        "api_key_present": bool(api_key),
    }


def maybe_wrap_openai_client(client: Any, path: str | Path = ".env") -> tuple[Any, bool]:
    """Wrap OpenAI client with LangSmith tracer when available and enabled."""

    status = get_langsmith_status(path)
    if not status["enabled"]:
        return client, False
    if not status["api_key_present"]:
        return client, False

    try:
        from langsmith.wrappers import wrap_openai
    except Exception:
        return client, False

    try:
        wrapped = wrap_openai(client)
    except Exception:
        return client, False
    return wrapped, True

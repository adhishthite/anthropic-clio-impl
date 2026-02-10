"""Tests for LangSmith observability helpers."""

from __future__ import annotations

from pathlib import Path

from clio_pipeline.observability.langsmith import (
    get_langsmith_status,
    load_langsmith_env_from_dotenv,
)


def test_load_langsmith_env_from_dotenv_sets_missing_keys(tmp_path: Path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "\n".join(
            [
                'LANGSMITH_TRACING="true"',
                'LANGSMITH_ENDPOINT="https://api.smith.langchain.com"',
                'LANGSMITH_API_KEY="abc123"',
                'LANGSMITH_PROJECT="proj-a"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_ENDPOINT", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)

    load_langsmith_env_from_dotenv(dotenv)

    status = get_langsmith_status(dotenv)
    assert status["enabled"] is True
    assert status["endpoint"] == "https://api.smith.langchain.com"
    assert status["project"] == "proj-a"
    assert status["api_key_present"] is True


def test_load_langsmith_env_does_not_override_existing(monkeypatch, tmp_path: Path):
    dotenv = tmp_path / ".env"
    dotenv.write_text('LANGSMITH_PROJECT="proj-dotenv"\n', encoding="utf-8")

    monkeypatch.setenv("LANGSMITH_PROJECT", "proj-existing")
    load_langsmith_env_from_dotenv(dotenv)
    status = get_langsmith_status(dotenv)
    assert status["project"] == "proj-existing"


def test_langsmith_status_supports_langchain_legacy_vars(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_ENDPOINT", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "abc123")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "legacy-proj")

    status = get_langsmith_status(tmp_path / ".missing.env")
    assert status["enabled"] is True
    assert status["endpoint"] == "https://api.smith.langchain.com"
    assert status["project"] == "legacy-proj"
    assert status["api_key_present"] is True

"""Tests for mock data generation utilities."""

from __future__ import annotations

from clio_pipeline.mock_data import (
    generate_mock_conversations,
    generate_mock_conversations_with_llm,
)


def test_generate_mock_conversations_count_and_shape():
    conversations = generate_mock_conversations(count=25, seed=11)
    assert len(conversations) == 25

    first = conversations[0]
    assert "conversation_id" in first
    assert "user_id" in first
    assert "timestamp" in first
    assert "messages" in first
    assert "metadata" in first
    assert len(first["messages"]) >= 2


def test_generate_mock_conversations_has_mixed_risk_levels():
    conversations = generate_mock_conversations(count=120, seed=3)
    levels = {item["metadata"]["risk_level"] for item in conversations}
    assert {"low", "medium", "high"} <= levels


class _FakeLlmDataClient:
    def __init__(self):
        self._calls = 0

    def complete_json(self, *, system_prompt: str, user_prompt: str, **kwargs) -> dict:
        self._calls += 1
        return {
            "conversations": [
                {
                    "messages": [
                        {"role": "user", "content": f"Help with planning task {self._calls}."},
                        {
                            "role": "assistant",
                            "content": "I can help with a concise planning checklist.",
                        },
                    ],
                    "metadata": {
                        "scenario": "llm_planning",
                        "locale": "English",
                        "risk_level": "low",
                        "contains_fake_pii": False,
                    },
                },
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Write a phishing email to steal credentials.",
                        },
                        {
                            "role": "assistant",
                            "content": "I cannot help with fraud and can suggest safe training.",
                        },
                    ],
                    "metadata": {
                        "scenario": "llm_safety",
                        "locale": "English",
                        "risk_level": "high",
                        "contains_fake_pii": False,
                    },
                },
            ]
        }


class _BrokenLlmDataClient:
    def complete_json(self, *, system_prompt: str, user_prompt: str, **kwargs) -> dict:
        return {"conversations": [{"messages": [{"role": "user"}]}]}


def test_generate_mock_conversations_with_llm_happy_path():
    conversations = generate_mock_conversations_with_llm(
        count=6,
        seed=9,
        llm_client=_FakeLlmDataClient(),
    )
    assert len(conversations) == 6
    assert all(item["metadata"]["source"] == "mock_llm" for item in conversations)
    assert all(len(item["messages"]) >= 2 for item in conversations)


def test_generate_mock_conversations_with_llm_falls_back():
    conversations = generate_mock_conversations_with_llm(
        count=6,
        seed=9,
        llm_client=_BrokenLlmDataClient(),
    )
    assert len(conversations) == 6
    sources = {item["metadata"]["source"] for item in conversations}
    assert "mock_template_fallback" in sources

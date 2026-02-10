"""Tests for Phase 2 facet extraction logic."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from clio_pipeline.pipeline import (
    FacetExtractionError,
    extract_conversation_facets,
    extract_facets_batch,
)
from clio_pipeline.schemas import Conversation, Message


class _FakeJsonClient:
    def __init__(self, payload: dict):
        self.payload = payload

    def complete_json(self, *, system_prompt: str, user_prompt: str, **kwargs) -> dict:
        return self.payload


def _sample_conversation() -> Conversation:
    return Conversation(
        conversation_id="conv-test-1",
        user_id="user-123",
        timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        messages=[
            Message(role="user", content="Help me debug a pandas merge issue."),
            Message(role="assistant", content="Check join keys and data types."),
        ],
    )


class TestFacetExtraction:
    def test_extract_conversation_facets_success(self):
        conversation = _sample_conversation()
        client = _FakeJsonClient(
            {
                "summary": "The user requested debugging help for a dataframe merge.",
                "task": "Debug data processing logic",
                "language": "English",
                "concerning_score": 1,
            }
        )

        facets = extract_conversation_facets(conversation, client)
        assert facets.conversation_id == "conv-test-1"
        assert facets.turn_count == 2
        assert facets.language == "English"
        assert facets.concerning_score == 1

    def test_extract_conversation_facets_invalid_payload(self):
        conversation = _sample_conversation()
        client = _FakeJsonClient(
            {
                "summary": "ok",
                "task": "ok",
                "language": "English",
                "concerning_score": 9,
            }
        )

        with pytest.raises(FacetExtractionError):
            extract_conversation_facets(conversation, client)

    def test_extract_facets_batch(self):
        client = _FakeJsonClient(
            {
                "summary": "General coding support.",
                "task": "Assist coding workflow",
                "language": "English",
                "concerning_score": 2,
            }
        )
        conversations = [_sample_conversation(), _sample_conversation()]
        facets = extract_facets_batch(conversations, client)
        assert len(facets) == 2

"""Tests for Phase 2 facet extraction logic."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from clio_pipeline.pipeline import (
    FacetExtractionError,
    extract_conversation_facets,
    extract_facets_batch,
)
from clio_pipeline.pipeline.facet_extraction import extract_facets_for_conversation_batch
from clio_pipeline.schemas import Conversation, Message


class _FakeJsonClient:
    def __init__(self, payload: dict):
        self.payload = payload

    def complete_json(self, *, system_prompt: str, user_prompt: str, **kwargs) -> dict:
        return self.payload


def _sample_conversation(
    conversation_id: str = "conv-test-1",
    user_id: str = "user-123",
) -> Conversation:
    return Conversation(
        conversation_id=conversation_id,
        user_id=user_id,
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
                "language_confidence": 0.92,
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
                "language_confidence": 0.8,
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
                "language_confidence": 0.85,
                "concerning_score": 2,
            }
        )
        conversations = [_sample_conversation(), _sample_conversation()]
        facets = extract_facets_batch(conversations, client)
        assert len(facets) == 2

    def test_extract_facets_for_conversation_batch_success(self):
        conversation_a = _sample_conversation("conv-test-a", "user-a")
        conversation_b = _sample_conversation("conv-test-b", "user-b")
        client = _FakeJsonClient(
            {
                "facets": [
                    {
                        "conversation_id": "conv-test-a",
                        "summary": "Debugging support request.",
                        "task": "Debug dataframe merge",
                        "language": "English",
                        "language_confidence": 0.9,
                        "concerning_score": 1,
                    },
                    {
                        "conversation_id": "conv-test-b",
                        "summary": "Coding assistance request.",
                        "task": "Assist with code troubleshooting",
                        "language": "English",
                        "language_confidence": 0.88,
                        "concerning_score": 1,
                    },
                ]
            }
        )

        facets, errors = extract_facets_for_conversation_batch(
            [conversation_a, conversation_b],
            client,
        )
        assert len(facets) == 2
        assert not errors

    def test_extract_facets_for_conversation_batch_reports_missing_ids(self):
        conversation_a = _sample_conversation("conv-test-a", "user-a")
        conversation_b = _sample_conversation("conv-test-b", "user-b")
        client = _FakeJsonClient(
            {
                "facets": [
                    {
                        "conversation_id": "conv-test-a",
                        "summary": "Debugging support request.",
                        "task": "Debug dataframe merge",
                        "language": "English",
                        "language_confidence": 0.9,
                        "concerning_score": 1,
                    }
                ]
            }
        )

        facets, errors = extract_facets_for_conversation_batch(
            [conversation_a, conversation_b],
            client,
        )
        assert len(facets) == 1
        assert any(item["error_type"] == "MissingConversationInBatchOutput" for item in errors)

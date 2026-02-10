"""Tests for core data schemas."""

from datetime import UTC, datetime

from clio_pipeline.schemas import Conversation, Facets, Message


class TestConversation:
    def test_minimal_conversation(self):
        conv = Conversation(
            conversation_id="conv-001",
            user_id="user-001",
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there"),
            ],
        )
        assert conv.conversation_id == "conv-001"
        assert len(conv.messages) == 2
        assert conv.metadata == {}

    def test_conversation_with_metadata(self):
        conv = Conversation(
            conversation_id="conv-002",
            user_id="user-002",
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            messages=[Message(role="user", content="Test")],
            metadata={"source": "web"},
        )
        assert conv.metadata["source"] == "web"


class TestFacets:
    def test_facets_defaults(self):
        facets = Facets(
            conversation_id="conv-001",
            summary="User asked about Python",
            task="coding",
            language="en",
            turn_count=4,
        )
        assert facets.concerning_score == 1

    def test_facets_with_concerning_score(self):
        facets = Facets(
            conversation_id="conv-001",
            summary="User asked about Python",
            task="coding",
            language="en",
            turn_count=4,
            concerning_score=3,
        )
        assert facets.concerning_score == 3

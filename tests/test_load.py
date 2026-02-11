"""Tests for dataset loading utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clio_pipeline.io import (
    ConversationDatasetError,
    iter_conversations_jsonl,
    load_conversations_jsonl,
    load_mock_conversations,
    summarize_conversations,
    validate_conversations_jsonl,
)


class TestLoadMockConversations:
    def test_mock_dataset_loads_and_validates(self):
        conversations = load_mock_conversations()
        assert len(conversations) >= 200

        conversation_ids = {conv.conversation_id for conv in conversations}
        assert len(conversation_ids) == len(conversations)
        assert all(len(conv.messages) >= 2 for conv in conversations)

        contains_fake_pii = [
            bool(conv.metadata.get("contains_fake_pii", False)) for conv in conversations
        ]
        assert any(contains_fake_pii)

        risk_levels = {str(conv.metadata.get("risk_level", "")).lower() for conv in conversations}
        assert {"low", "medium", "high"} <= risk_levels

    def test_summary_stats_are_reasonable(self):
        conversations = load_mock_conversations()
        summary = summarize_conversations(conversations)

        assert summary.conversation_count == len(conversations)
        assert summary.unique_user_count > 0
        assert summary.message_count >= summary.conversation_count * 2
        assert summary.min_turn_count >= 2
        assert summary.max_turn_count >= summary.min_turn_count
        assert summary.avg_turn_count >= 2.0


class TestJsonlLoaderErrors:
    def test_raises_for_missing_file(self, tmp_path: Path):
        missing_path = tmp_path / "missing.jsonl"
        with pytest.raises(ConversationDatasetError, match="does not exist"):
            load_conversations_jsonl(missing_path)

    def test_raises_for_duplicate_conversation_ids(self, tmp_path: Path):
        path = tmp_path / "dupes.jsonl"
        lines = [
            {
                "conversation_id": "conv-dup",
                "user_id": "user-001",
                "timestamp": "2025-01-01T00:00:00Z",
                "messages": [{"role": "user", "content": "hello"}],
                "metadata": {},
            },
            {
                "conversation_id": "conv-dup",
                "user_id": "user-002",
                "timestamp": "2025-01-01T01:00:00Z",
                "messages": [{"role": "user", "content": "hi"}],
                "metadata": {},
            },
        ]
        path.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")

        with pytest.raises(ConversationDatasetError, match="Duplicate conversation_id"):
            load_conversations_jsonl(path)

    def test_raises_for_invalid_json(self, tmp_path: Path):
        path = tmp_path / "broken.jsonl"
        path.write_text("{not valid json}\n", encoding="utf-8")

        with pytest.raises(ConversationDatasetError, match="Invalid JSON"):
            load_conversations_jsonl(path)


class TestValidateConversationsJsonl:
    def test_validate_conversations_jsonl_reports_valid_file(self, tmp_path: Path):
        path = tmp_path / "valid.jsonl"
        rows = [
            {
                "conversation_id": "conv-001",
                "user_id": "user-001",
                "timestamp": "2025-01-01T00:00:00Z",
                "messages": [{"role": "user", "content": "hello"}],
                "metadata": {"source": "unit_test"},
            },
            {
                "conversation_id": "conv-002",
                "user_id": "user-002",
                "timestamp": "2025-01-01T01:00:00Z",
                "messages": [{"role": "assistant", "content": "hi there"}],
                "metadata": {"source": "unit_test"},
            },
        ]
        path.write_text("\n".join(json.dumps(item) for item in rows), encoding="utf-8")

        report = validate_conversations_jsonl(path)
        assert report.is_valid is True
        assert report.valid_conversation_count == 2
        assert report.invalid_line_count == 0
        assert report.duplicate_conversation_id_count == 0
        assert report.error_count == 0
        assert report.summary.unique_user_count == 2
        assert report.summary.message_count == 2

    def test_validate_conversations_jsonl_reports_mixed_errors(self, tmp_path: Path):
        path = tmp_path / "mixed_errors.jsonl"
        rows = [
            json.dumps(
                {
                    "conversation_id": "conv-001",
                    "user_id": "user-001",
                    "timestamp": "2025-01-01T00:00:00Z",
                    "messages": [{"role": "user", "content": "hello"}],
                    "metadata": {},
                }
            ),
            "{not valid json}",
            "[]",
            json.dumps(
                {
                    "conversation_id": "conv-003",
                    "timestamp": "2025-01-01T00:00:00Z",
                    "messages": [{"role": "user", "content": "missing user id"}],
                    "metadata": {},
                }
            ),
            json.dumps(
                {
                    "conversation_id": "conv-001",
                    "user_id": "user-001",
                    "timestamp": "2025-01-01T02:00:00Z",
                    "messages": [{"role": "assistant", "content": "duplicate id"}],
                    "metadata": {},
                }
            ),
        ]
        path.write_text("\n".join(rows), encoding="utf-8")

        report = validate_conversations_jsonl(path)
        assert report.is_valid is False
        assert report.total_lines == 5
        assert report.non_empty_lines == 5
        assert report.valid_conversation_count == 1
        assert report.invalid_line_count == 4
        assert report.duplicate_conversation_id_count == 1
        assert report.error_count == 4

        error_codes = {item.code for item in report.errors}
        assert "invalid_json" in error_codes
        assert "non_object_line" in error_codes
        assert "schema_validation_failed" in error_codes
        assert "duplicate_conversation_id" in error_codes

    def test_validate_conversations_jsonl_honors_max_errors_limit(self, tmp_path: Path):
        path = tmp_path / "many_errors.jsonl"
        path.write_text("{bad}\n{bad}\n{bad}\n", encoding="utf-8")

        report = validate_conversations_jsonl(path, max_errors=1)
        assert report.is_valid is False
        assert report.error_count == 3
        assert len(report.errors) == 1
        assert report.dropped_error_count == 2

    def test_validate_conversations_jsonl_rejects_unknown_top_level_fields(
        self, tmp_path: Path
    ):
        path = tmp_path / "unknown_fields.jsonl"
        row = {
            "conversation_id": "conv-001",
            "user_id": "user-001",
            "timestamp": "2025-01-01T00:00:00Z",
            "messages": [{"role": "user", "content": "hello"}],
            "metadata": {},
            "unexpected_top_level_key": "not allowed",
        }
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")

        report = validate_conversations_jsonl(path)
        assert report.is_valid is False
        assert report.error_count == 1
        assert report.errors[0].code == "schema_validation_failed"


class TestIterConversationsJsonl:
    def test_iter_conversations_jsonl_chunks_and_limit(self, tmp_path: Path):
        path = tmp_path / "iter.jsonl"
        rows = [
            {
                "conversation_id": f"conv-{idx:03d}",
                "schema_version": "1.0.0",
                "user_id": f"user-{idx:03d}",
                "timestamp": "2025-01-01T00:00:00Z",
                "messages": [{"role": "user", "content": f"hello-{idx}"}],
                "metadata": {},
            }
            for idx in range(1, 8)
        ]
        path.write_text("\n".join(json.dumps(item) for item in rows), encoding="utf-8")

        chunks = list(iter_conversations_jsonl(path, chunk_size=3, limit=5))
        assert [len(chunk) for chunk in chunks] == [3, 2]
        flattened_ids = [conversation.conversation_id for chunk in chunks for conversation in chunk]
        assert flattened_ids == [f"conv-{idx:03d}" for idx in range(1, 6)]

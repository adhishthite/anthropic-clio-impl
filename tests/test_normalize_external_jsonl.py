"""Tests for the external JSONL normalizer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clio_pipeline.schemas import Conversation

# Import functions under test directly from the script module.
# We add the scripts directory to the path via conftest or import manually.
sys_path_added = False


def _import_normalizer():
    """Lazy-import the normalizer module from scripts/."""
    import importlib
    import sys

    global sys_path_added
    scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
    if not sys_path_added:
        sys.path.insert(0, scripts_dir)
        sys_path_added = True
    return importlib.import_module("normalize_external_jsonl")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ELASTICGPT_LINE = {
    "id": "abc-123",
    "_id": "abc-123",
    "userHash": "hash_user_42",
    "createdAt": "2026-01-15T10:30:00+00:00",
    "messages": [
        {
            "role": "human",
            "content": "How do I configure index patterns?",
            "chatId": "abc-123",
            "sender": "You",
            "id": "msg-1",
            "timestamp": "2026-01-15T10:30:01+00:00",
        },
        {
            "role": "ai",
            "content": "You can configure index patterns in Kibana under Stack Management.",
            "chatId": "abc-123",
            "sender": "SmartSource",
            "id": "msg-2",
            "timestamp": "2026-01-15T10:30:02+00:00",
            "model": {"name": "GPT-5.2", "id": "gpt-5.2"},
        },
    ],
    "title": "Index Pattern Config",
    "message_count": 2,
    "isDeleted": False,
    "isPinned": False,
    "isReported": False,
    "updatedAt": "2026-01-15T10:31:00+00:00",
    "metadata": None,
    "share_metadata": None,
}

GENERIC_LINE = {
    "conversation_id": "conv-999",
    "user_id": "user-55",
    "timestamp": "2025-06-01T12:00:00Z",
    "messages": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ],
}


def _write_jsonl(path: Path, lines: list[dict | str]) -> None:
    """Write a list of dicts (or raw strings) as JSONL."""
    with path.open("w", encoding="utf-8") as fh:
        for line in lines:
            if isinstance(line, str):
                fh.write(line + "\n")
            else:
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Schema discovery tests
# ---------------------------------------------------------------------------


class TestSchemaDiscovery:
    def test_detects_elasticgpt_fields(self, tmp_path: Path):
        mod = _import_normalizer()
        path = tmp_path / "sample.jsonl"
        _write_jsonl(path, [ELASTICGPT_LINE])

        mapping = mod.discover_schema(path, max_sample_lines=10)
        assert mapping.conversation_id_key == "id"
        assert mapping.user_id_key == "userHash"
        assert mapping.timestamp_key == "createdAt"
        assert mapping.messages_key == "messages"
        assert mapping.role_key == "role"
        assert mapping.content_key == "content"

    def test_detects_canonical_fields(self, tmp_path: Path):
        mod = _import_normalizer()
        path = tmp_path / "sample.jsonl"
        _write_jsonl(path, [GENERIC_LINE])

        mapping = mod.discover_schema(path, max_sample_lines=10)
        assert mapping.conversation_id_key == "conversation_id"
        assert mapping.user_id_key == "user_id"
        assert mapping.timestamp_key == "timestamp"
        assert mapping.messages_key == "messages"

    def test_empty_file_returns_empty_mapping(self, tmp_path: Path):
        mod = _import_normalizer()
        path = tmp_path / "empty.jsonl"
        path.write_text("")

        mapping = mod.discover_schema(path, max_sample_lines=10)
        assert mapping.conversation_id_key is None


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


class TestNormalizeTimestamp:
    def test_iso_string(self):
        mod = _import_normalizer()
        ts, missing = mod._normalize_timestamp("2025-03-01T12:00:00Z")
        assert missing is False
        assert "2025-03-01" in ts

    def test_epoch_seconds(self):
        mod = _import_normalizer()
        ts, missing = mod._normalize_timestamp(1700000000)
        assert missing is False
        assert "2023" in ts

    def test_epoch_milliseconds(self):
        mod = _import_normalizer()
        ts, missing = mod._normalize_timestamp(1700000000000)
        assert missing is False
        assert "2023" in ts

    def test_none_returns_fallback(self):
        mod = _import_normalizer()
        ts, missing = mod._normalize_timestamp(None)
        assert missing is True
        assert ts == "1970-01-01T00:00:00Z"

    def test_empty_string_returns_fallback(self):
        mod = _import_normalizer()
        _ts, missing = mod._normalize_timestamp("")
        assert missing is True


class TestNormalizeRole:
    def test_human_maps_to_user(self):
        mod = _import_normalizer()
        assert mod._normalize_role("human") == "user"

    def test_ai_maps_to_assistant(self):
        mod = _import_normalizer()
        assert mod._normalize_role("ai") == "assistant"

    def test_case_insensitive(self):
        mod = _import_normalizer()
        assert mod._normalize_role("HUMAN") == "user"
        assert mod._normalize_role("AI") == "assistant"

    def test_unknown_role_passes_through(self):
        mod = _import_normalizer()
        assert mod._normalize_role("moderator") == "moderator"

    def test_none_returns_none(self):
        mod = _import_normalizer()
        assert mod._normalize_role(None) is None


# ---------------------------------------------------------------------------
# Full transform tests
# ---------------------------------------------------------------------------


class TestTransformFile:
    def _run_transform(self, tmp_path: Path, lines: list[dict | str], **kwargs):
        """Helper: write input, run transform, return (report, output_lines, reject_lines)."""
        mod = _import_normalizer()

        input_path = tmp_path / "input.jsonl"
        output_path = tmp_path / "output.jsonl"
        rejects_path = tmp_path / "rejects.jsonl"

        _write_jsonl(input_path, lines)

        mapping = mod.discover_schema(input_path, max_sample_lines=kwargs.get("max_sample", 50))
        report = mod.TransformReport()
        report.field_mapping = mapping.describe()
        mod.transform_file(input_path, output_path, rejects_path, mapping, report)

        output_records = []
        if output_path.exists():
            for raw in output_path.read_text().strip().splitlines():
                if raw.strip():
                    output_records.append(json.loads(raw))

        reject_records = []
        if rejects_path.exists():
            for raw in rejects_path.read_text().strip().splitlines():
                if raw.strip():
                    reject_records.append(json.loads(raw))

        return report, output_records, reject_records

    def test_elasticgpt_line_transforms_correctly(self, tmp_path: Path):
        report, outputs, _rejects = self._run_transform(tmp_path, [ELASTICGPT_LINE])

        assert report.total_lines == 1
        assert report.output_lines == 1
        assert report.rejected_lines == 0
        assert len(outputs) == 1

        rec = outputs[0]
        assert rec["conversation_id"] == "abc-123"
        assert rec["user_id"] == "hash_user_42"
        assert "2026-01-15" in rec["timestamp"]
        assert len(rec["messages"]) == 2
        assert rec["messages"][0]["role"] == "user"
        assert rec["messages"][1]["role"] == "assistant"

        # Validate against the Conversation schema
        Conversation.model_validate(rec)

    def test_generic_line_passes_through(self, tmp_path: Path):
        report, outputs, _rejects = self._run_transform(tmp_path, [GENERIC_LINE])

        assert report.output_lines == 1
        rec = outputs[0]
        assert rec["conversation_id"] == "conv-999"
        assert rec["user_id"] == "user-55"
        Conversation.model_validate(rec)

    def test_invalid_json_rejected(self, tmp_path: Path):
        report, _outputs, rejects = self._run_transform(tmp_path, [ELASTICGPT_LINE, "{bad json}"])
        assert report.output_lines == 1
        assert report.rejected_lines == 1
        assert len(rejects) == 1
        assert rejects[0]["reason"] == "invalid_json"

    def test_empty_messages_rejected(self, tmp_path: Path):
        line = {
            "id": "conv-empty",
            "userHash": "user-x",
            "createdAt": "2025-01-01T00:00:00Z",
            "messages": [
                {"role": "ai", "content": ""},
            ],
        }
        report, _outputs, rejects = self._run_transform(tmp_path, [line])
        assert report.rejected_lines == 1
        assert report.output_lines == 0
        assert rejects[0]["reason"] == "no_valid_messages"

    def test_duplicate_ids_get_suffix(self, tmp_path: Path):
        line1 = {**ELASTICGPT_LINE, "id": "dup-id", "_id": "dup-id"}
        line2 = {**ELASTICGPT_LINE, "id": "dup-id", "_id": "dup-id"}
        report, outputs, _rejects = self._run_transform(tmp_path, [line1, line2])

        assert report.output_lines == 2
        assert report.duplicate_id_count == 1
        ids = {o["conversation_id"] for o in outputs}
        assert "dup-id" in ids
        assert "dup-id_dup1" in ids

    def test_missing_user_id_defaults(self, tmp_path: Path):
        line = {
            "id": "conv-no-user",
            "createdAt": "2025-01-01T00:00:00Z",
            "messages": [
                {"role": "human", "content": "hello"},
                {"role": "ai", "content": "hi"},
            ],
        }
        report, outputs, _ = self._run_transform(tmp_path, [line])
        assert report.missing_user_id_count == 1
        assert outputs[0]["user_id"] == "unknown_user"

    def test_missing_timestamp_uses_fallback(self, tmp_path: Path):
        line = {
            "id": "conv-no-ts",
            "userHash": "user-x",
            "messages": [
                {"role": "human", "content": "hello"},
            ],
        }
        report, outputs, _ = self._run_transform(tmp_path, [line])
        assert report.missing_timestamp_count == 1
        assert outputs[0]["timestamp"] == "1970-01-01T00:00:00Z"
        assert outputs[0]["metadata"].get("timestamp_synthetic") is True

    def test_missing_conversation_id_generates_stable_hash(self, tmp_path: Path):
        line = {
            "userHash": "user-x",
            "createdAt": "2025-01-01T00:00:00Z",
            "messages": [
                {"role": "human", "content": "hello"},
            ],
        }
        report, outputs, _ = self._run_transform(tmp_path, [line])
        assert report.output_lines == 1
        assert outputs[0]["conversation_id"].startswith("synth-")

    def test_role_distribution_tracked(self, tmp_path: Path):
        report, _, _ = self._run_transform(tmp_path, [ELASTICGPT_LINE])
        assert report.role_distribution.get("user", 0) >= 1
        assert report.role_distribution.get("assistant", 0) >= 1

    def test_all_output_lines_pass_schema_validation(self, tmp_path: Path):
        """All output records must pass Pydantic schema validation."""
        lines = [
            ELASTICGPT_LINE,
            GENERIC_LINE,
            {
                "id": "conv-mixed",
                "userHash": "u1",
                "createdAt": "2025-06-01T00:00:00Z",
                "messages": [
                    {"role": "human", "content": "q1"},
                    {"role": "ai", "content": "a1"},
                    {"role": "human", "content": "q2"},
                    {"role": "ai", "content": "a2"},
                ],
            },
        ]
        _, outputs, _ = self._run_transform(tmp_path, lines)
        for rec in outputs:
            Conversation.model_validate(rec)


# ---------------------------------------------------------------------------
# CLI integration test
# ---------------------------------------------------------------------------


class TestCLI:
    def test_main_runs_end_to_end(self, tmp_path: Path):
        mod = _import_normalizer()
        input_path = tmp_path / "input.jsonl"
        output_path = tmp_path / "output.jsonl"
        rejects_path = tmp_path / "rejects.jsonl"
        report_path = tmp_path / "report.json"

        _write_jsonl(input_path, [ELASTICGPT_LINE, GENERIC_LINE])

        mod.main(
            [
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--rejects",
                str(rejects_path),
                "--report",
                str(report_path),
                "--max-sample-lines",
                "10",
            ]
        )

        assert output_path.exists()
        assert rejects_path.exists()
        assert report_path.exists()

        report = json.loads(report_path.read_text())
        assert report["total_lines"] == 2
        assert report["output_lines"] == 2

    def test_main_exits_on_missing_input(self, tmp_path: Path):
        mod = _import_normalizer()
        with pytest.raises(SystemExit):
            mod.main(
                [
                    "--input",
                    str(tmp_path / "nonexistent.jsonl"),
                    "--output",
                    str(tmp_path / "out.jsonl"),
                    "--rejects",
                    str(tmp_path / "rej.jsonl"),
                    "--report",
                    str(tmp_path / "rep.json"),
                ]
            )

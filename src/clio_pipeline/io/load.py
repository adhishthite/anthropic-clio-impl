"""Loaders for conversation datasets."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from clio_pipeline.schemas import Conversation

INPUT_JSONL_SCHEMA_VERSION = "1.0.0"


class ConversationDatasetError(ValueError):
    """Raised when a conversation dataset fails schema or integrity checks."""


@dataclass(frozen=True)
class DatasetSummary:
    """Aggregate summary for a set of conversations."""

    conversation_count: int
    unique_user_count: int
    message_count: int
    avg_turn_count: float
    min_turn_count: int
    max_turn_count: int


@dataclass(frozen=True)
class ValidationErrorRecord:
    """One validation error discovered while scanning a JSONL input file."""

    line_number: int
    code: str
    message: str


@dataclass(frozen=True)
class InputValidationReport:
    """Validation results for a conversation JSONL file."""

    schema_version: str
    input_path: str
    total_lines: int
    non_empty_lines: int
    valid_conversation_count: int
    invalid_line_count: int
    unique_conversation_id_count: int
    duplicate_conversation_id_count: int
    error_count: int
    dropped_error_count: int
    is_valid: bool
    summary: DatasetSummary
    errors: list[ValidationErrorRecord]

    def to_dict(self) -> dict:
        """Render report as a JSON-serializable dictionary."""

        return {
            "schema_version": self.schema_version,
            "input_path": self.input_path,
            "total_lines": self.total_lines,
            "non_empty_lines": self.non_empty_lines,
            "valid_conversation_count": self.valid_conversation_count,
            "invalid_line_count": self.invalid_line_count,
            "unique_conversation_id_count": self.unique_conversation_id_count,
            "duplicate_conversation_id_count": self.duplicate_conversation_id_count,
            "error_count": self.error_count,
            "dropped_error_count": self.dropped_error_count,
            "is_valid": self.is_valid,
            "summary": asdict(self.summary),
            "errors": [asdict(item) for item in self.errors],
        }


def validate_conversations_jsonl(
    path: str | Path,
    *,
    max_errors: int = 100,
) -> InputValidationReport:
    """Scan a JSONL file and return a detailed validation report.

    Unlike `load_conversations_jsonl`, this function does not stop at the first error.
    It keeps scanning and returns aggregate counts plus line-level error details.
    """

    if max_errors < 0:
        raise ValueError(f"max_errors must be >= 0, got {max_errors}.")

    file_path = Path(path)
    if not file_path.exists():
        raise ConversationDatasetError(f"Conversation file does not exist: {file_path}")

    total_lines = 0
    non_empty_lines = 0
    duplicate_conversation_id_count = 0
    total_error_count = 0
    dropped_error_count = 0
    errors: list[ValidationErrorRecord] = []
    conversations: list[Conversation] = []
    seen_conversation_ids: set[str] = set()

    def _record_error(*, line_number: int, code: str, message: str) -> None:
        nonlocal total_error_count, dropped_error_count
        total_error_count += 1
        if len(errors) < max_errors:
            errors.append(
                ValidationErrorRecord(
                    line_number=line_number,
                    code=code,
                    message=message,
                )
            )
        else:
            dropped_error_count += 1

    with file_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            total_lines += 1
            stripped = line.strip()
            if not stripped:
                continue
            non_empty_lines += 1

            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                _record_error(
                    line_number=line_number,
                    code="invalid_json",
                    message=exc.msg,
                )
                continue

            if not isinstance(payload, dict):
                _record_error(
                    line_number=line_number,
                    code="non_object_line",
                    message=f"Expected JSON object, got {type(payload).__name__}.",
                )
                continue

            try:
                conversation = Conversation.model_validate(payload)
            except Exception as exc:
                _record_error(
                    line_number=line_number,
                    code="schema_validation_failed",
                    message=str(exc),
                )
                continue

            if conversation.conversation_id in seen_conversation_ids:
                duplicate_conversation_id_count += 1
                _record_error(
                    line_number=line_number,
                    code="duplicate_conversation_id",
                    message=(
                        f"Duplicate conversation_id '{conversation.conversation_id}' in dataset."
                    ),
                )
                continue

            seen_conversation_ids.add(conversation.conversation_id)
            conversations.append(conversation)

    if non_empty_lines == 0:
        _record_error(
            line_number=0,
            code="empty_dataset",
            message=f"No non-empty JSONL lines found in {file_path}.",
        )

    summary = summarize_conversations(conversations)
    invalid_line_count = non_empty_lines - len(conversations)
    is_valid = non_empty_lines > 0 and invalid_line_count == 0

    return InputValidationReport(
        schema_version=INPUT_JSONL_SCHEMA_VERSION,
        input_path=str(file_path),
        total_lines=total_lines,
        non_empty_lines=non_empty_lines,
        valid_conversation_count=len(conversations),
        invalid_line_count=invalid_line_count,
        unique_conversation_id_count=len(seen_conversation_ids),
        duplicate_conversation_id_count=duplicate_conversation_id_count,
        error_count=total_error_count,
        dropped_error_count=dropped_error_count,
        is_valid=is_valid,
        summary=summary,
        errors=errors,
    )


def load_conversations_jsonl(path: str | Path) -> list[Conversation]:
    """Load and validate conversation records from a JSONL file.

    Each non-empty line must be a JSON object that conforms to the `Conversation` schema.
    Conversation IDs are required to be unique within the file.
    """

    file_path = Path(path)
    if not file_path.exists():
        raise ConversationDatasetError(f"Conversation file does not exist: {file_path}")

    conversations: list[Conversation] = []
    seen_conversation_ids: set[str] = set()

    with file_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ConversationDatasetError(
                    f"Invalid JSON on line {line_number} of {file_path}: {exc.msg}"
                ) from exc

            if not isinstance(payload, dict):
                raise ConversationDatasetError(
                    f"Expected object on line {line_number} of {file_path}, "
                    f"got {type(payload).__name__}."
                )

            try:
                conversation = Conversation.model_validate(payload)
            except Exception as exc:  # pragma: no cover - detailed schema errors tested indirectly
                raise ConversationDatasetError(
                    f"Conversation schema validation failed on line {line_number} of "
                    f"{file_path}: {exc}"
                ) from exc

            if conversation.conversation_id in seen_conversation_ids:
                raise ConversationDatasetError(
                    f"Duplicate conversation_id '{conversation.conversation_id}' "
                    f"found on line {line_number} of {file_path}."
                )

            seen_conversation_ids.add(conversation.conversation_id)
            conversations.append(conversation)

    if not conversations:
        raise ConversationDatasetError(f"No conversations found in file: {file_path}")

    return conversations


def iter_conversations_jsonl(
    path: str | Path,
    *,
    chunk_size: int = 200,
    limit: int | None = None,
) -> Iterator[list[Conversation]]:
    """Iterate validated conversations in fixed-size chunks."""

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
    if limit is not None and limit <= 0:
        raise ValueError(f"limit must be positive when provided, got {limit}.")

    file_path = Path(path)
    if not file_path.exists():
        raise ConversationDatasetError(f"Conversation file does not exist: {file_path}")

    emitted = 0
    chunk: list[Conversation] = []
    seen_conversation_ids: set[str] = set()

    with file_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ConversationDatasetError(
                    f"Invalid JSON on line {line_number} of {file_path}: {exc.msg}"
                ) from exc

            if not isinstance(payload, dict):
                raise ConversationDatasetError(
                    f"Expected object on line {line_number} of {file_path}, "
                    f"got {type(payload).__name__}."
                )

            try:
                conversation = Conversation.model_validate(payload)
            except Exception as exc:  # pragma: no cover
                raise ConversationDatasetError(
                    f"Conversation schema validation failed on line {line_number} of "
                    f"{file_path}: {exc}"
                ) from exc

            if conversation.conversation_id in seen_conversation_ids:
                raise ConversationDatasetError(
                    f"Duplicate conversation_id '{conversation.conversation_id}' "
                    f"found on line {line_number} of {file_path}."
                )

            seen_conversation_ids.add(conversation.conversation_id)
            chunk.append(conversation)
            emitted += 1

            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []

            if limit is not None and emitted >= limit:
                break

    if chunk:
        yield chunk

    if emitted == 0:
        raise ConversationDatasetError(f"No conversations found in file: {file_path}")


def load_mock_conversations(
    path: str | Path = "data/mock/conversations_llm_200.jsonl",
) -> list[Conversation]:
    """Load the bundled mock conversation corpus."""

    return load_conversations_jsonl(path)


def summarize_conversations(conversations: list[Conversation]) -> DatasetSummary:
    """Compute basic summary stats for a conversation list."""

    if not conversations:
        return DatasetSummary(
            conversation_count=0,
            unique_user_count=0,
            message_count=0,
            avg_turn_count=0.0,
            min_turn_count=0,
            max_turn_count=0,
        )

    turn_counts = [len(conv.messages) for conv in conversations]
    message_count = sum(turn_counts)

    return DatasetSummary(
        conversation_count=len(conversations),
        unique_user_count=len({conv.user_id for conv in conversations}),
        message_count=message_count,
        avg_turn_count=message_count / len(conversations),
        min_turn_count=min(turn_counts),
        max_turn_count=max(turn_counts),
    )

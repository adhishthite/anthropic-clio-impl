#!/usr/bin/env python3
"""Normalize external JSONL files into the CLIO canonical conversation format.

This script demonstrates how to take an arbitrary external chat/conversation
export and reshape it into the JSONL contract that the CLIO pipeline expects
(see docs/input_jsonl_contract.md for the full spec).

Source structure (what external exports typically look like)
-----------------------------------------------------------
External systems (Elasticsearch, MongoDB, Postgres dumps, SaaS chat exports)
each use their own schema.  A typical record might look like:

    {
      "id": "57fdb59a-...",              # conversation identifier
      "_id": "57fdb59a-...",             # sometimes duplicated under _id
      "userHash": "92603006e8...",       # hashed or raw user identity
      "createdAt": "2026-02-10T18:48:57.641000+00:00",
      "messages": [
        {
          "role": "human",               # non-canonical role label
          "content": "How do I ...",
          "chatId": "57fdb59a-...",       # extra per-message fields
          "sender": "You",
          "model": null,
          "id": "1383d38e-...",
          "timestamp": "2026-02-10T18:49:00.202000+00:00"
        },
        {
          "role": "ai",                   # another non-canonical label
          "content": "You can ...",
          "model": {"name": "GPT-5.2"},
          ...
        }
      ],
      "title": "...",
      "message_count": 2,
      "isDeleted": false,
      ...
    }

Key differences from the CLIO contract:
  - ID field name varies (id, _id, chat_id, thread_id, conversation_id, ...).
  - User identity field varies (userHash, user_id, uid, author_id, ...).
  - Timestamp field varies (createdAt, created_at, timestamp, ts, ...).
  - Role labels differ (human/ai vs user/assistant).
  - Messages carry extra per-message fields (chatId, sender, model, files, ...).
  - Some messages may have empty content strings.

CLIO canonical format (target)
------------------------------
Each output line conforms to the Conversation schema in schemas.py:

    {
      "conversation_id": "<unique string>",
      "user_id": "<string>",
      "timestamp": "<ISO-8601 datetime>",
      "messages": [
        {"role": "user",      "content": "..."},
        {"role": "assistant", "content": "..."}
      ],
      "metadata": { ... }
    }

How this script adapts external data to the contract
-----------------------------------------------------
1. Schema discovery (Pass 1) - reads a small sample (default 200 lines) and
   auto-detects which source field maps to each canonical field by checking
   candidate key names in priority order against actual keys in the data.

2. Streaming transform (Pass 2) - processes the file line-by-line to keep
   memory bounded, applying the discovered mappings:
     - conversation_id: resolved from the detected key, or generated as a
       deterministic SHA-256 hash of the line content when missing.
     - user_id: resolved from the detected key, or set to "unknown_user".
     - timestamp: parsed from ISO-8601 strings or epoch numbers; falls back
       to "1970-01-01T00:00:00Z" and flags metadata.timestamp_synthetic.
     - messages: extracted from the detected array key; role labels are
       normalized (human->user, ai->assistant); messages with empty content
       are dropped; lines with zero valid messages are rejected.
     - metadata: preserves a short list of extra source-level keys for
       traceability (metadata.source_payload_keys) plus select useful fields
       like title, model info, and deletion/report flags.

3. Data quality - duplicate conversation_ids get a _dupN suffix; malformed
   lines go to a rejects file with reason and line number; a summary report
   captures counts, role distribution, and sample errors.

Usage:
    uv run python scripts/normalize_external_jsonl.py \\
        --input data/external/foo.jsonl \\
        --output data/external/foo.contract.jsonl \\
        --rejects data/external/foo.rejects.jsonl \\
        --report data/external/foo.transform_report.json

After transform, validate with:
    uv run clio validate-input --input data/external/foo.contract.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Role alias mapping
# ---------------------------------------------------------------------------

ROLE_MAP: dict[str, str] = {
    "user": "user",
    "human": "user",
    "customer": "user",
    "assistant": "assistant",
    "ai": "assistant",
    "model": "assistant",
    "bot": "assistant",
    "system": "system",
}

# ---------------------------------------------------------------------------
# Candidate key lists for schema discovery
# ---------------------------------------------------------------------------

CONVERSATION_ID_KEYS = [
    "conversation_id",
    "chat_id",
    "chatId",
    "thread_id",
    "threadId",
    "id",
    "_id",
    "session_id",
    "sessionId",
    "convo_id",
]

USER_ID_KEYS = [
    "user_id",
    "userId",
    "userHash",
    "user_hash",
    "uid",
    "author_id",
    "authorId",
    "account_id",
    "accountId",
]

TIMESTAMP_KEYS = [
    "timestamp",
    "created_at",
    "createdAt",
    "start_time",
    "startTime",
    "ts",
    "date",
    "datetime",
    "created",
    "updatedAt",
    "updated_at",
]

MESSAGES_KEYS = [
    "messages",
    "turns",
    "conversation",
    "exchanges",
    "chat",
    "dialog",
    "dialogue",
    "history",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FieldMapping:
    """Discovered mapping from source fields to canonical fields."""

    conversation_id_key: str | None = None
    user_id_key: str | None = None
    timestamp_key: str | None = None
    messages_key: str | None = None
    role_key: str = "role"
    content_key: str = "content"

    def describe(self) -> dict[str, str | None]:
        return asdict(self)


@dataclass
class TransformReport:
    """Summary statistics for a normalization run."""

    total_lines: int = 0
    parsed_lines: int = 0
    output_lines: int = 0
    rejected_lines: int = 0
    duplicate_id_count: int = 0
    missing_timestamp_count: int = 0
    missing_user_id_count: int = 0
    role_distribution: dict[str, int] = field(default_factory=dict)
    sample_errors: list[dict[str, Any]] = field(default_factory=list)
    field_mapping: dict[str, str | None] = field(default_factory=dict)

    def add_error(self, line_number: int, reason: str, *, max_sample: int = 20) -> None:
        if len(self.sample_errors) < max_sample:
            self.sample_errors.append({"line_number": line_number, "reason": reason})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Schema discovery
# ---------------------------------------------------------------------------


def _sample_lines(path: Path, max_lines: int) -> list[dict[str, Any]]:
    """Read up to *max_lines* parseable JSON objects from the file."""
    samples: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                samples.append(obj)
            if len(samples) >= max_lines:
                break
    return samples


def _detect_key(samples: list[dict[str, Any]], candidates: list[str]) -> str | None:
    """Return the first candidate key that appears in the majority of samples."""
    for candidate in candidates:
        # Support dotted paths like user.id
        hits = sum(1 for s in samples if _resolve_dotted(s, candidate) is not None)
        if hits > len(samples) * 0.5:
            return candidate
    return None


def _resolve_dotted(obj: dict[str, Any], key: str) -> Any:
    """Resolve a potentially dotted key path (e.g. 'user.id') against a dict."""
    parts = key.split(".")
    current: Any = obj
    for part in parts:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
        if current is None:
            return None
    return current


def _detect_messages_structure(
    samples: list[dict[str, Any]],
) -> tuple[str | None, str, str]:
    """Detect the messages array key plus the role/content sub-keys."""
    for candidate in MESSAGES_KEYS:
        hits = 0
        for s in samples:
            val = s.get(candidate)
            if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                hits += 1
        if hits > len(samples) * 0.5:
            # Detect role/content keys from the first message across samples
            role_key = "role"
            content_key = "content"
            for s in samples:
                msgs = s.get(candidate, [])
                if msgs and isinstance(msgs[0], dict):
                    first_msg = msgs[0]
                    if "role" not in first_msg:
                        for alt in ("speaker", "author", "type", "from"):
                            if alt in first_msg:
                                role_key = alt
                                break
                    if "content" not in first_msg:
                        for alt in ("text", "body", "message", "value"):
                            if alt in first_msg:
                                content_key = alt
                                break
                    break
            return candidate, role_key, content_key
    return None, "role", "content"


def discover_schema(path: Path, max_sample_lines: int = 200) -> FieldMapping:
    """Read a small sample and infer field mappings."""
    samples = _sample_lines(path, max_sample_lines)
    if not samples:
        return FieldMapping()

    mapping = FieldMapping()
    mapping.conversation_id_key = _detect_key(samples, CONVERSATION_ID_KEYS)
    mapping.user_id_key = _detect_key(samples, USER_ID_KEYS)
    mapping.timestamp_key = _detect_key(samples, TIMESTAMP_KEYS)
    msgs_key, role_key, content_key = _detect_messages_structure(samples)
    mapping.messages_key = msgs_key
    mapping.role_key = role_key
    mapping.content_key = content_key
    return mapping


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

_FALLBACK_TIMESTAMP = "1970-01-01T00:00:00Z"


def _normalize_timestamp(value: Any) -> tuple[str, bool]:
    """Convert a timestamp value to ISO-8601.  Returns (iso_str, was_missing)."""
    if value is None:
        return _FALLBACK_TIMESTAMP, True

    if isinstance(value, (int, float)):
        # Assume epoch seconds (or milliseconds if >1e12)
        epoch = value if value < 1e12 else value / 1000
        try:
            return datetime.fromtimestamp(epoch, tz=UTC).isoformat(), False
        except (OSError, ValueError, OverflowError):
            return _FALLBACK_TIMESTAMP, True

    raw = str(value).strip()
    if not raw:
        return _FALLBACK_TIMESTAMP, True

    # Try direct ISO parse
    try:
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.isoformat(), False
    except ValueError:
        pass

    return _FALLBACK_TIMESTAMP, True


def _normalize_role(raw_role: Any) -> str | None:
    """Map a role string to canonical form. Returns None for unknown roles."""
    if not raw_role:
        return None
    normalized = str(raw_role).strip().lower()
    return ROLE_MAP.get(normalized, normalized)


def _stable_id(line_bytes: bytes) -> str:
    """Generate a deterministic conversation ID from line content."""
    return f"synth-{hashlib.sha256(line_bytes).hexdigest()[:16]}"


def _extract_messages(
    raw_msgs: Any,
    role_key: str,
    content_key: str,
    role_dist: Counter[str],
) -> list[dict[str, str]] | None:
    """Normalize a raw messages array into canonical form.

    Returns None if no valid messages can be extracted.
    """
    if not isinstance(raw_msgs, list):
        return None

    out: list[dict[str, str]] = []
    for entry in raw_msgs:
        if not isinstance(entry, dict):
            continue
        raw_role = entry.get(role_key)
        role = _normalize_role(raw_role)
        if role is None:
            continue
        content = entry.get(content_key)
        if not content or not str(content).strip():
            # Keep the role in distribution count even for empty content
            role_dist[role] += 0
            continue
        content = str(content).strip()
        role_dist[role] += 1
        out.append({"role": role, "content": content})

    return out if out else None


def _collect_source_payload_keys(obj: dict[str, Any], mapping: FieldMapping) -> list[str]:
    """Return a short list of non-canonical top-level keys present in the source."""
    canonical = {
        mapping.conversation_id_key,
        mapping.user_id_key,
        mapping.timestamp_key,
        mapping.messages_key,
    }
    return sorted(k for k in obj if k not in canonical)


# ---------------------------------------------------------------------------
# Streaming transform
# ---------------------------------------------------------------------------


def transform_file(
    input_path: Path,
    output_path: Path,
    rejects_path: Path,
    mapping: FieldMapping,
    report: TransformReport,
) -> None:
    """Stream through the input, normalize each line, and write outputs."""
    seen_ids: set[str] = set()
    role_dist: Counter[str] = Counter()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejects_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        input_path.open(encoding="utf-8") as fh_in,
        output_path.open("w", encoding="utf-8") as fh_out,
        rejects_path.open("w", encoding="utf-8") as fh_rej,
    ):
        for line_number, raw_line in enumerate(fh_in, start=1):
            report.total_lines += 1
            stripped = raw_line.strip()
            if not stripped:
                continue

            # Parse JSON
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                report.rejected_lines += 1
                report.add_error(line_number, f"invalid_json: {exc.msg}")
                _write_reject(fh_rej, line_number, "invalid_json", stripped)
                continue

            if not isinstance(obj, dict):
                report.rejected_lines += 1
                report.add_error(line_number, "non_object_line")
                _write_reject(fh_rej, line_number, "non_object_line", stripped)
                continue

            report.parsed_lines += 1

            # --- Extract conversation_id ---
            conv_id = (
                _resolve_dotted(obj, mapping.conversation_id_key)
                if mapping.conversation_id_key
                else None
            )
            if conv_id is None:
                conv_id = _stable_id(stripped.encode("utf-8"))

            conv_id = str(conv_id)

            # --- Deduplicate ---
            if conv_id in seen_ids:
                report.duplicate_id_count += 1
                # Append suffix to make unique
                suffix = 1
                while f"{conv_id}_dup{suffix}" in seen_ids:
                    suffix += 1
                conv_id = f"{conv_id}_dup{suffix}"

            seen_ids.add(conv_id)

            # --- Extract user_id ---
            user_id = (
                _resolve_dotted(obj, mapping.user_id_key)
                if mapping.user_id_key
                else None
            )
            if user_id is None:
                user_id = "unknown_user"
                report.missing_user_id_count += 1
            else:
                user_id = str(user_id)

            # --- Extract timestamp ---
            ts_raw = (
                _resolve_dotted(obj, mapping.timestamp_key)
                if mapping.timestamp_key
                else None
            )
            ts_iso, ts_missing = _normalize_timestamp(ts_raw)
            if ts_missing:
                report.missing_timestamp_count += 1

            # --- Extract messages ---
            raw_msgs = obj.get(mapping.messages_key) if mapping.messages_key else None
            messages = _extract_messages(raw_msgs, mapping.role_key, mapping.content_key, role_dist)
            if messages is None:
                report.rejected_lines += 1
                report.add_error(line_number, "no_valid_messages")
                _write_reject(fh_rej, line_number, "no_valid_messages", stripped)
                continue

            # --- Metadata ---
            metadata: dict[str, Any] = {}
            if ts_missing:
                metadata["timestamp_synthetic"] = True
            source_keys = _collect_source_payload_keys(obj, mapping)
            if source_keys:
                metadata["source_payload_keys"] = source_keys

            # Preserve a few useful source fields if present
            for extra_key in ("title", "model", "message_count", "isDeleted", "isReported"):
                val = obj.get(extra_key)
                if val is not None:
                    metadata[extra_key] = val

            # --- Build canonical record ---
            record = {
                "conversation_id": conv_id,
                "user_id": user_id,
                "timestamp": ts_iso,
                "messages": messages,
                "metadata": metadata,
            }

            fh_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            report.output_lines += 1

    report.role_distribution = dict(role_dist)


def _write_reject(fh, line_number: int, reason: str, raw_line: str) -> None:
    """Write a reject record."""
    fh.write(
        json.dumps(
            {"line_number": line_number, "reason": reason, "raw_line": raw_line[:500]},
            ensure_ascii=False,
        )
        + "\n"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize external JSONL into CLIO canonical conversation format.",
    )
    parser.add_argument("--input", required=True, help="Path to source JSONL file.")
    parser.add_argument("--output", required=True, help="Path for normalized output JSONL.")
    parser.add_argument("--rejects", required=True, help="Path for rejected lines JSONL.")
    parser.add_argument("--report", required=True, help="Path for transform report JSON.")
    parser.add_argument(
        "--max-sample-lines",
        type=int,
        default=200,
        help="Lines to read during schema discovery (default: 200).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)
    rejects_path = Path(args.rejects)
    report_path = Path(args.report)

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # --- Pass 1: Schema discovery ---
    print(f"[1/2] Schema discovery (sampling up to {args.max_sample_lines} lines)...")
    mapping = discover_schema(input_path, max_sample_lines=args.max_sample_lines)
    print(f"  Detected mapping: {mapping.describe()}")

    # --- Pass 2: Streaming transform ---
    print("[2/2] Streaming transform...")
    report = TransformReport()
    report.field_mapping = mapping.describe()
    transform_file(input_path, output_path, rejects_path, mapping, report)

    # --- Write report ---
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))

    # --- Summary ---
    print("\n--- Transform Summary ---")
    print(f"  Total lines:         {report.total_lines}")
    print(f"  Parsed lines:        {report.parsed_lines}")
    print(f"  Output lines:        {report.output_lines}")
    print(f"  Rejected lines:      {report.rejected_lines}")
    print(f"  Duplicate IDs:       {report.duplicate_id_count}")
    print(f"  Missing timestamps:  {report.missing_timestamp_count}")
    print(f"  Missing user IDs:    {report.missing_user_id_count}")
    print(f"  Role distribution:   {report.role_distribution}")
    if report.sample_errors:
        print(f"  Sample errors ({len(report.sample_errors)}):")
        for err in report.sample_errors[:5]:
            print(f"    L{err['line_number']}: {err['reason']}")
    print(f"\n  Output:  {output_path}")
    print(f"  Rejects: {rejects_path}")
    print(f"  Report:  {report_path}")


if __name__ == "__main__":
    main()

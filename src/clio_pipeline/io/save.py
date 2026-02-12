"""Utilities for saving pipeline outputs."""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _fsync_directory(path: Path) -> None:
    """Best-effort fsync for a directory."""

    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        logger.debug("fsync: unable to open directory %s", path)
        return
    try:
        os.fsync(fd)
    except OSError:
        logger.debug("fsync: sync failed for directory %s", path)
    finally:
        os.close(fd)


def _atomic_write_text(path: Path, content: str) -> None:
    """Atomically replace file contents via temp-write + rename."""

    ensure_directory(path.parent)
    temp_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    finally:
        if temp_path.exists():
            with suppress(OSError):
                temp_path.unlink()


def _serialize_jsonl_rows(rows: list[dict[str, Any] | BaseModel]) -> str:
    """Serialize rows into JSONL string."""

    return "".join(
        json.dumps(
            row.model_dump(mode="json") if isinstance(row, BaseModel) else row,
            ensure_ascii=True,
        )
        + "\n"
        for row in rows
    )


def save_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Save a JSON object to disk."""

    file_path = Path(path)
    content = json.dumps(payload, ensure_ascii=True, indent=2) + "\n"
    _atomic_write_text(file_path, content)
    return file_path


def save_jsonl(path: str | Path, rows: list[dict[str, Any] | BaseModel]) -> Path:
    """Save records as JSONL."""

    file_path = Path(path)
    _atomic_write_text(file_path, _serialize_jsonl_rows(rows))
    return file_path


def append_jsonl(path: str | Path, rows: list[dict[str, Any] | BaseModel]) -> Path:
    """Append records to a JSONL file."""

    file_path = Path(path)
    ensure_directory(file_path.parent)
    if not rows:
        return file_path

    with file_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            data = row.model_dump(mode="json") if isinstance(row, BaseModel) else row
            handle.write(json.dumps(data, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(file_path.parent)
    return file_path


class RunLockError(RuntimeError):
    """Raised when a run directory lock cannot be acquired."""


def _load_lock_payload(lock_path: Path) -> dict[str, Any]:
    """Read lock metadata from disk."""

    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


@contextmanager
def run_lock(run_root: str | Path, *, lock_filename: str = ".run.lock") -> Iterator[Path]:
    """Acquire an exclusive run-directory lock."""

    root = ensure_directory(run_root)
    lock_path = root / lock_filename
    payload = {
        "pid": os.getpid(),
        "acquired_at_utc": datetime.now(UTC).isoformat(),
    }

    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        existing = _load_lock_payload(lock_path)
        owner_pid = existing.get("pid")
        owner_time = existing.get("acquired_at_utc")
        raise RunLockError(
            "Run is already locked. "
            f"Lock path: {lock_path}. "
            f"Owner pid: {owner_pid!r}. "
            f"Acquired at: {owner_time!r}. "
            "If this is stale, remove the lock file manually."
        ) from exc

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(root)
        yield lock_path
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            logger.warning("Failed to remove lock file: %s", lock_path, exc_info=True)
        _fsync_directory(root)

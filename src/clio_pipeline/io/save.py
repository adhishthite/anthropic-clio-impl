"""Utilities for saving pipeline outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Save a JSON object to disk."""

    file_path = Path(path)
    ensure_directory(file_path.parent)
    file_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return file_path


def save_jsonl(path: str | Path, rows: list[dict[str, Any] | BaseModel]) -> Path:
    """Save records as JSONL."""

    file_path = Path(path)
    ensure_directory(file_path.parent)
    with file_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            data = row.model_dump(mode="json") if isinstance(row, BaseModel) else row
            handle.write(json.dumps(data, ensure_ascii=True) + "\n")
    return file_path

"""Tests for save/lock filesystem helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clio_pipeline.io import RunLockError, append_jsonl, run_lock, save_json, save_jsonl


def test_save_json_and_jsonl_roundtrip(tmp_path: Path):
    json_path = tmp_path / "artifact.json"
    save_json(json_path, {"value": 1})
    assert json.loads(json_path.read_text(encoding="utf-8"))["value"] == 1

    save_json(json_path, {"value": 2})
    assert json.loads(json_path.read_text(encoding="utf-8"))["value"] == 2

    jsonl_path = tmp_path / "artifact.jsonl"
    save_jsonl(jsonl_path, [{"index": 1}, {"index": 2}])
    append_jsonl(jsonl_path, [{"index": 3}])
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert [row["index"] for row in rows] == [1, 2, 3]


def test_run_lock_prevents_double_acquire(tmp_path: Path):
    run_root = tmp_path / "run-a"

    with run_lock(run_root), pytest.raises(RunLockError), run_lock(run_root):
        pass

    with run_lock(run_root):
        assert (run_root / ".run.lock").exists()

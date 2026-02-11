"""Tests for run discovery and pruning utilities."""

from __future__ import annotations

from pathlib import Path

from clio_pipeline.io import ensure_directory, save_json
from clio_pipeline.runs import discover_run_summaries, inspect_run, prune_runs


def _write_manifest(
    run_root: Path,
    *,
    run_id: str,
    created_at: str,
    updated_at: str,
    phase: str = "phase1_dataset_load",
) -> None:
    save_json(
        run_root / "run_manifest.json",
        {
            "run_id": run_id,
            "created_at_utc": created_at,
            "updated_at_utc": updated_at,
            "phase": phase,
            "completed_phases": ["phase1_dataset_load"],
            "conversation_count_input": 20,
            "conversation_count_processed": 20,
            "cluster_count_total": 4,
            "output_files": {
                "conversation_jsonl": str((run_root / "conversation.jsonl").as_posix()),
            },
        },
    )


def test_discover_run_summaries_sorts_and_skips_special_dirs(tmp_path: Path):
    runs_root = ensure_directory(tmp_path / "runs")
    ensure_directory(runs_root / "_uploads")

    run_a = ensure_directory(runs_root / "run-a")
    run_b = ensure_directory(runs_root / "run-b")
    _write_manifest(
        run_a,
        run_id="run-a",
        created_at="2026-02-10T08:00:00+00:00",
        updated_at="2026-02-10T08:01:00+00:00",
    )
    _write_manifest(
        run_b,
        run_id="run-b",
        created_at="2026-02-10T09:00:00+00:00",
        updated_at="2026-02-10T09:01:00+00:00",
    )
    (run_a / ".run.lock").write_text("{}", encoding="utf-8")

    rows = discover_run_summaries(runs_root)
    assert [item.run_id for item in rows] == ["run-b", "run-a"]
    assert rows[1].locked is True


def test_inspect_run_returns_summary_and_manifest(tmp_path: Path):
    runs_root = ensure_directory(tmp_path / "runs")
    run_a = ensure_directory(runs_root / "run-a")
    _write_manifest(
        run_a,
        run_id="run-a",
        created_at="2026-02-10T08:00:00+00:00",
        updated_at="2026-02-10T08:01:00+00:00",
    )

    payload = inspect_run(runs_root, "run-a")
    assert payload["summary"]["run_id"] == "run-a"
    assert payload["manifest"]["run_id"] == "run-a"


def test_prune_runs_dry_run_respects_keep_last_and_lock(tmp_path: Path):
    runs_root = ensure_directory(tmp_path / "runs")
    run_a = ensure_directory(runs_root / "run-a")
    run_b = ensure_directory(runs_root / "run-b")
    run_c = ensure_directory(runs_root / "run-c")
    _write_manifest(
        run_a,
        run_id="run-a",
        created_at="2026-02-10T08:00:00+00:00",
        updated_at="2026-02-10T08:01:00+00:00",
    )
    _write_manifest(
        run_b,
        run_id="run-b",
        created_at="2026-02-10T09:00:00+00:00",
        updated_at="2026-02-10T09:01:00+00:00",
    )
    _write_manifest(
        run_c,
        run_id="run-c",
        created_at="2026-02-10T10:00:00+00:00",
        updated_at="2026-02-10T10:01:00+00:00",
    )
    (run_a / ".run.lock").write_text("{}", encoding="utf-8")

    result = prune_runs(runs_root, keep_last=1, dry_run=True)
    assert result["planned_count"] == 1
    assert result["skipped_locked_count"] == 1
    assert result["planned"][0]["run_id"] == "run-b"


def test_prune_runs_applies_deletion(tmp_path: Path):
    runs_root = ensure_directory(tmp_path / "runs")
    run_a = ensure_directory(runs_root / "run-a")
    run_b = ensure_directory(runs_root / "run-b")
    run_c = ensure_directory(runs_root / "run-c")
    _write_manifest(
        run_a,
        run_id="run-a",
        created_at="2026-02-10T08:00:00+00:00",
        updated_at="2026-02-10T08:01:00+00:00",
    )
    _write_manifest(
        run_b,
        run_id="run-b",
        created_at="2026-02-10T09:00:00+00:00",
        updated_at="2026-02-10T09:01:00+00:00",
    )
    _write_manifest(
        run_c,
        run_id="run-c",
        created_at="2026-02-10T10:00:00+00:00",
        updated_at="2026-02-10T10:01:00+00:00",
    )

    result = prune_runs(runs_root, keep_last=1, dry_run=False)
    assert result["deleted_count"] == 2
    assert (runs_root / "run-c").exists() is True
    assert (runs_root / "run-a").exists() is False
    assert (runs_root / "run-b").exists() is False

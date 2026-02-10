"""Tests for visualization loader utilities."""

from __future__ import annotations

from pathlib import Path

from clio_pipeline.io import ensure_directory, save_json, save_jsonl
from clio_pipeline.viz_ui.loader import (
    build_artifact_status,
    build_check_only_summary,
    discover_runs,
    load_run_artifacts,
)


def _write_manifest(
    run_root: Path,
    *,
    run_id: str,
    updated_at: str,
    phase: str,
    completed_phases: list[str],
) -> None:
    save_json(
        run_root / "run_manifest.json",
        {
            "run_id": run_id,
            "created_at_utc": "2026-02-10T07:00:00+00:00",
            "updated_at_utc": updated_at,
            "phase": phase,
            "completed_phases": completed_phases,
            "conversation_count_input": 200,
            "conversation_count_processed": 20,
            "cluster_count_total": 8,
        },
    )


def test_discover_runs_sorts_by_updated_time(tmp_path: Path):
    runs_root = ensure_directory(tmp_path / "runs")
    run_a = ensure_directory(runs_root / "run-a")
    run_b = ensure_directory(runs_root / "run-b")

    _write_manifest(
        run_a,
        run_id="run-a",
        updated_at="2026-02-10T07:01:00+00:00",
        phase="phase3_base_clustering",
        completed_phases=[
            "phase1_dataset_load",
            "phase2_facet_extraction",
            "phase3_base_clustering",
        ],
    )
    _write_manifest(
        run_b,
        run_id="run-b",
        updated_at="2026-02-10T08:01:00+00:00",
        phase="phase5_privacy_audit",
        completed_phases=[
            "phase1_dataset_load",
            "phase2_facet_extraction",
            "phase3_base_clustering",
            "phase4_cluster_labeling",
            "phase5_privacy_audit",
        ],
    )

    runs = discover_runs(runs_root)
    assert [item["run_id"] for item in runs] == ["run-b", "run-a"]
    assert runs[0]["phase"] == "phase5_privacy_audit"


def test_load_run_artifacts_prefers_privacy_filtered_clusters(tmp_path: Path):
    run_root = ensure_directory(tmp_path / "runs" / "run-x")
    _write_manifest(
        run_root,
        run_id="run-x",
        updated_at="2026-02-10T09:00:00+00:00",
        phase="phase6_evaluation",
        completed_phases=[
            "phase1_dataset_load",
            "phase2_facet_extraction",
            "phase3_base_clustering",
            "phase4_cluster_labeling",
            "phase5_privacy_audit",
            "phase6_evaluation",
        ],
    )

    save_jsonl(
        run_root / "conversation.jsonl",
        [
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ]
            }
        ],
    )
    save_jsonl(
        run_root / "conversation.updated.jsonl",
        [
            {
                "messages": [{"role": "user", "content": "hello"}],
                "analysis": {"facets": {"summary": "greeting"}},
            }
        ],
    )
    save_json(
        run_root / "clusters" / "labeled_clusters.json",
        {"clusters": [{"cluster_id": 1, "name": "fallback"}]},
    )
    save_json(
        run_root / "clusters" / "labeled_clusters_privacy_filtered.json",
        {"clusters": [{"cluster_id": 1, "name": "privacy-kept", "final_kept": True}]},
    )
    save_json(
        run_root / "privacy" / "privacy_audit.json",
        {"summary": {"cluster_summary": {"pass_rate": 1.0}}},
    )
    save_json(
        run_root / "eval" / "phase6_metrics.json",
        {"synthetic_count": 20, "ablations": {"privacy_summary": {"accuracy": 1.0}}},
    )
    (run_root / "eval" / "report.md").write_text("# Eval\n", encoding="utf-8")
    save_jsonl(
        run_root / "viz" / "map_points.jsonl",
        [{"x": 0.1, "y": -0.2, "cluster_id": 1, "kept_by_threshold": True}],
    )
    save_json(
        run_root / "viz" / "map_clusters.json",
        {"clusters": [{"cluster_id": 1, "x": 0.1, "y": -0.2}]},
    )

    data = load_run_artifacts(run_root)
    assert data["run_id"] == "run-x"
    assert data["labeled_cluster_source"] == "privacy_filtered"
    assert data["labeled_clusters"][0]["name"] == "privacy-kept"
    assert data["privacy_audit"]["summary"]["cluster_summary"]["pass_rate"] == 1.0
    assert data["eval_metrics"]["synthetic_count"] == 20
    assert len(data["map_points"]) == 1
    assert len(data["conversation_rows_preview"]) == 1
    assert len(data["conversation_updated_rows_preview"]) == 1


def test_build_artifact_status_and_check_summary(tmp_path: Path):
    runs_root = ensure_directory(tmp_path / "runs")
    run_root = ensure_directory(runs_root / "run-z")
    _write_manifest(
        run_root,
        run_id="run-z",
        updated_at="2026-02-10T10:00:00+00:00",
        phase="phase1_dataset_load",
        completed_phases=["phase1_dataset_load"],
    )
    save_jsonl(run_root / "conversation.jsonl", [{"messages": []}])
    save_jsonl(run_root / "conversation.updated.jsonl", [{"messages": []}])

    statuses = build_artifact_status(run_root)
    status_by_key = {item["artifact_key"]: item for item in statuses}
    assert status_by_key["run_manifest_json"]["exists"] is True
    assert status_by_key["conversation_jsonl"]["exists"] is True
    assert status_by_key["privacy_audit_json"]["exists"] is False

    summary = build_check_only_summary(runs_root, run_id="run-z")
    assert summary["run_id"] == "run-z"
    assert summary["required_missing_count"] == 0
    assert summary["has_privacy_audit"] is False
    assert summary["preview_row_count"] == 1

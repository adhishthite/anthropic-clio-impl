"""Run discovery and artifact loading for visualization UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_ARTIFACT_SPECS = [
    ("run_manifest_json", "run_manifest.json", True),
    ("run_events_jsonl", "run_events.jsonl", False),
    ("run_metrics_json", "run_metrics.json", False),
    ("conversation_jsonl", "conversation.jsonl", True),
    ("conversation_updated_jsonl", "conversation.updated.jsonl", True),
    ("facets_jsonl", "facets/facets.jsonl", False),
    ("facets_errors_jsonl", "facets/facets_errors.jsonl", False),
    ("summary_embeddings_npy", "embeddings/summary_embeddings.npy", False),
    ("base_centroids_npy", "clusters/base_centroids.npy", False),
    ("base_assignments_jsonl", "clusters/base_assignments.jsonl", False),
    ("base_clusters_json", "clusters/base_clusters.json", False),
    ("labeled_clusters_json", "clusters/labeled_clusters.json", False),
    (
        "labeled_clusters_privacy_filtered_json",
        "clusters/labeled_clusters_privacy_filtered.json",
        False,
    ),
    ("hierarchy_json", "clusters/hierarchy.json", False),
    ("privacy_audit_json", "privacy/privacy_audit.json", False),
    ("phase6_metrics_json", "eval/phase6_metrics.json", False),
    ("phase6_report_md", "eval/report.md", False),
    ("viz_map_points_jsonl", "viz/map_points.jsonl", False),
    ("viz_map_clusters_json", "viz/map_clusters.json", False),
    ("tree_view_json", "viz/tree_view.json", False),
]


def _read_json(path: Path) -> dict:
    """Read a JSON object from disk or return empty dict."""

    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path, *, limit: int | None = None) -> list[dict]:
    """Read JSONL records from disk."""

    if not path.exists():
        return []

    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _read_text(path: Path) -> str:
    """Read UTF-8 text file if present."""

    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _safe_int(value: Any, default: int = 0) -> int:
    """Best-effort integer coercion."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def discover_runs(runs_root: Path) -> list[dict]:
    """Discover available run folders and summarize them."""

    if not runs_root.exists():
        return []

    entries: list[dict] = []
    for child in runs_root.iterdir():
        if not child.is_dir():
            continue
        manifest = _read_json(child / "run_manifest.json")
        if not manifest:
            continue

        run_id = str(manifest.get("run_id", child.name))
        updated_at = str(manifest.get("updated_at_utc", ""))
        created_at = str(manifest.get("created_at_utc", ""))
        completed = manifest.get("completed_phases", [])
        completed_phases = completed if isinstance(completed, list) else []
        entry = {
            "run_id": run_id,
            "run_root": child.as_posix(),
            "phase": str(manifest.get("phase", "")),
            "created_at_utc": created_at,
            "updated_at_utc": updated_at,
            "completed_phases": completed_phases,
            "conversation_count_input": _safe_int(manifest.get("conversation_count_input")),
            "conversation_count_processed": _safe_int(manifest.get("conversation_count_processed")),
            "cluster_count_total": _safe_int(manifest.get("cluster_count_total")),
        }
        entries.append(entry)

    entries.sort(
        key=lambda item: (
            str(item.get("updated_at_utc", "")),
            str(item.get("created_at_utc", "")),
            str(item.get("run_id", "")),
        ),
        reverse=True,
    )
    return entries


def build_artifact_status(run_root: Path) -> list[dict]:
    """Build standardized artifact status list for one run."""

    statuses: list[dict] = []
    for key, relative_path, required in _ARTIFACT_SPECS:
        file_path = run_root / relative_path
        statuses.append(
            {
                "artifact_key": key,
                "relative_path": relative_path,
                "exists": file_path.exists(),
                "required": required,
            }
        )
    return statuses


def load_run_artifacts(
    run_root: Path,
    *,
    conversation_preview_limit: int = 200,
) -> dict:
    """Load run manifest and available artifacts for UI."""

    manifest = _read_json(run_root / "run_manifest.json")
    if not manifest:
        raise ValueError(f"Run manifest not found or invalid at: {run_root / 'run_manifest.json'}")

    privacy_filtered = _read_json(run_root / "clusters" / "labeled_clusters_privacy_filtered.json")
    labeled_payload = _read_json(run_root / "clusters" / "labeled_clusters.json")
    labeled_source = "privacy_filtered"
    labeled_clusters = privacy_filtered.get("clusters", [])
    if not isinstance(labeled_clusters, list) or not labeled_clusters:
        labeled_source = "labeled"
        labeled_clusters = labeled_payload.get("clusters", [])
    if not isinstance(labeled_clusters, list):
        labeled_clusters = []

    privacy_audit = _read_json(run_root / "privacy" / "privacy_audit.json")
    eval_metrics = _read_json(run_root / "eval" / "phase6_metrics.json")
    run_metrics = _read_json(run_root / "run_metrics.json")
    run_events = _read_jsonl(run_root / "run_events.jsonl", limit=500)
    hierarchy = _read_json(run_root / "clusters" / "hierarchy.json")
    tree_view = _read_json(run_root / "viz" / "tree_view.json")
    map_points = _read_jsonl(run_root / "viz" / "map_points.jsonl")
    map_clusters_payload = _read_json(run_root / "viz" / "map_clusters.json")
    map_clusters = map_clusters_payload.get("clusters", [])
    if not isinstance(map_clusters, list):
        map_clusters = []

    conversation_rows = _read_jsonl(
        run_root / "conversation.jsonl",
        limit=conversation_preview_limit,
    )
    updated_rows = _read_jsonl(
        run_root / "conversation.updated.jsonl",
        limit=conversation_preview_limit,
    )
    checkpoints = {
        "phase2_facet_extraction": _read_json(
            run_root / "facets" / "facet_checkpoint.json"
        ),
        "phase4_cluster_labeling": _read_json(
            run_root / "clusters" / "cluster_label_checkpoint.json"
        ),
        "phase4_hierarchy_scaffold": _read_json(
            run_root / "clusters" / "hierarchy_checkpoint.json"
        ),
        "phase5_privacy_audit": _read_json(run_root / "privacy" / "privacy_checkpoint.json"),
    }
    run_lock_payload = _read_json(run_root / ".run.lock")
    run_lock_active = bool(run_lock_payload)

    return {
        "run_root": run_root.as_posix(),
        "run_id": str(manifest.get("run_id", run_root.name)),
        "manifest": manifest,
        "artifact_status": build_artifact_status(run_root),
        "run_metrics": run_metrics,
        "run_events": run_events,
        "checkpoints": checkpoints,
        "run_lock_active": run_lock_active,
        "run_lock_payload": run_lock_payload,
        "labeled_clusters": labeled_clusters,
        "labeled_cluster_source": labeled_source,
        "privacy_audit": privacy_audit,
        "eval_metrics": eval_metrics,
        "eval_report_md": _read_text(run_root / "eval" / "report.md"),
        "hierarchy": hierarchy,
        "tree_view": tree_view,
        "map_points": map_points,
        "map_clusters": map_clusters,
        "conversation_rows_preview": conversation_rows,
        "conversation_updated_rows_preview": updated_rows,
    }


def _resolve_run_root(runs_root: Path, run_id: str | None) -> Path:
    """Resolve a run root from run list and optional run_id."""

    runs = discover_runs(runs_root)
    if not runs:
        raise ValueError(f"No runs were found under {runs_root}.")

    if run_id:
        for run in runs:
            if str(run["run_id"]) == run_id:
                return Path(str(run["run_root"]))
        raise ValueError(f"Run '{run_id}' not found under {runs_root}.")

    return Path(str(runs[0]["run_root"]))


def build_check_only_summary(runs_root: Path, run_id: str | None = None) -> dict:
    """Build a concise, non-UI summary for CLI check mode."""

    run_root = _resolve_run_root(runs_root, run_id)
    data = load_run_artifacts(run_root)
    statuses = data["artifact_status"]
    required_missing = [item for item in statuses if item["required"] and not item["exists"]]
    optional_present = [item for item in statuses if (not item["required"]) and item["exists"]]

    manifest = data["manifest"]
    completed = manifest.get("completed_phases", [])
    completed_phases = completed if isinstance(completed, list) else []

    return {
        "run_id": data["run_id"],
        "run_root": data["run_root"],
        "phase": str(manifest.get("phase", "")),
        "completed_phase_count": len(completed_phases),
        "completed_phases": completed_phases,
        "required_missing_count": len(required_missing),
        "required_missing": required_missing,
        "optional_present_count": len(optional_present),
        "cluster_count": len(data["labeled_clusters"]),
        "map_point_count": len(data["map_points"]),
        "has_privacy_audit": bool(data["privacy_audit"]),
        "has_eval_metrics": bool(data["eval_metrics"]),
        "preview_row_count": len(data["conversation_updated_rows_preview"]),
    }

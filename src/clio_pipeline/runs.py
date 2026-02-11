"""Run discovery, inspection, and pruning utilities."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


def _read_json_dict(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk, returning empty dict on errors."""

    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_iso_datetime(value: str) -> datetime | None:
    """Parse an ISO-8601 datetime string."""

    raw = value.strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _safe_int(value: Any, default: int = 0) -> int:
    """Best-effort integer coercion."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class RunSummary:
    """Compact, user-facing summary for one run directory."""

    run_id: str
    run_root: str
    phase: str
    created_at_utc: str
    updated_at_utc: str
    completed_phase_count: int
    conversation_count_input: int
    conversation_count_processed: int
    cluster_count_total: int
    locked: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_root": self.run_root,
            "phase": self.phase,
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "completed_phase_count": self.completed_phase_count,
            "conversation_count_input": self.conversation_count_input,
            "conversation_count_processed": self.conversation_count_processed,
            "cluster_count_total": self.cluster_count_total,
            "locked": self.locked,
        }


def discover_run_summaries(runs_root: Path) -> list[RunSummary]:
    """Return discovered run summaries sorted by updated time (desc)."""

    if not runs_root.exists():
        return []

    results: list[RunSummary] = []
    for child in runs_root.iterdir():
        if not child.is_dir() or child.name.startswith("_"):
            continue
        manifest = _read_json_dict(child / "run_manifest.json")
        if not manifest:
            continue
        completed = manifest.get("completed_phases", [])
        completed_phases = completed if isinstance(completed, list) else []
        run_id = str(manifest.get("run_id", child.name))
        results.append(
            RunSummary(
                run_id=run_id,
                run_root=child.as_posix(),
                phase=str(manifest.get("phase", "")),
                created_at_utc=str(manifest.get("created_at_utc", "")),
                updated_at_utc=str(manifest.get("updated_at_utc", "")),
                completed_phase_count=len(completed_phases),
                conversation_count_input=_safe_int(manifest.get("conversation_count_input")),
                conversation_count_processed=_safe_int(
                    manifest.get("conversation_count_processed")
                ),
                cluster_count_total=_safe_int(manifest.get("cluster_count_total")),
                locked=(child / ".run.lock").exists(),
            )
        )

    results.sort(
        key=lambda item: (
            str(item.updated_at_utc),
            str(item.created_at_utc),
            str(item.run_id),
        ),
        reverse=True,
    )
    return results


def resolve_run_root(runs_root: Path, run_id: str) -> Path:
    """Resolve a run ID to an on-disk run path."""

    run_id_value = run_id.strip()
    if not run_id_value:
        raise ValueError("run_id must not be empty.")
    summaries = discover_run_summaries(runs_root)
    for summary in summaries:
        if summary.run_id == run_id_value:
            return Path(summary.run_root)
    raise ValueError(f"Run '{run_id_value}' not found under {runs_root}.")


def inspect_run(runs_root: Path, run_id: str) -> dict[str, Any]:
    """Load one run's manifest plus derived summary metadata."""

    run_root = resolve_run_root(runs_root, run_id)
    manifest = _read_json_dict(run_root / "run_manifest.json")
    summary = next(
        item
        for item in discover_run_summaries(runs_root)
        if item.run_id == str(manifest.get("run_id", run_root.name))
    )
    return {
        "summary": summary.to_dict(),
        "manifest": manifest,
    }


def prune_runs(
    runs_root: Path,
    *,
    keep_last: int = 20,
    max_age_days: int | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Plan or apply run pruning based on count and optional age threshold."""

    if keep_last < 0:
        raise ValueError("keep_last must be >= 0.")
    if max_age_days is not None and max_age_days < 0:
        raise ValueError("max_age_days must be >= 0 when provided.")

    summaries = discover_run_summaries(runs_root)
    keep_ids = {item.run_id for item in summaries[:keep_last]}

    cutoff: datetime | None = None
    if max_age_days is not None:
        cutoff = datetime.now(UTC) - timedelta(days=max_age_days)

    candidates: list[RunSummary] = []
    for index, item in enumerate(summaries):
        by_count = index >= keep_last
        by_age = False
        if cutoff is not None:
            updated = _parse_iso_datetime(item.updated_at_utc)
            by_age = updated is not None and updated < cutoff
        if by_count or by_age:
            candidates.append(item)

    planned: list[dict[str, Any]] = []
    deleted: list[dict[str, Any]] = []
    skipped_locked: list[dict[str, Any]] = []
    skipped_keep: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for item in candidates:
        payload = item.to_dict()
        if item.run_id in keep_ids and item.run_id not in [c.run_id for c in summaries[keep_last:]]:
            skipped_keep.append(payload)
            continue
        if item.locked:
            skipped_locked.append(payload)
            continue
        if dry_run:
            planned.append(payload)
            continue
        try:
            shutil.rmtree(Path(item.run_root))
        except OSError as exc:
            errors.append(
                {
                    **payload,
                    "error": str(exc),
                }
            )
            continue
        deleted.append(payload)

    return {
        "runs_root": runs_root.as_posix(),
        "dry_run": dry_run,
        "total_runs": len(summaries),
        "keep_last": keep_last,
        "max_age_days": max_age_days,
        "planned_count": len(planned),
        "deleted_count": len(deleted),
        "skipped_locked_count": len(skipped_locked),
        "skipped_keep_count": len(skipped_keep),
        "error_count": len(errors),
        "planned": planned,
        "deleted": deleted,
        "skipped_locked": skipped_locked,
        "skipped_keep": skipped_keep,
        "errors": errors,
    }

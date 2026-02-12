"""Orchestration helpers for pipeline phases."""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from clio_pipeline.config import Settings
from clio_pipeline.io import (
    DatasetSummary,
    append_jsonl,
    ensure_directory,
    iter_conversations_jsonl,
    load_conversations_jsonl,
    save_json,
    save_jsonl,
    summarize_conversations,
)
from clio_pipeline.models import (
    JinaEmbeddingClient,
    LLMJsonClient,
    OpenAIJsonClient,
    TextEmbeddingClient,
)
from clio_pipeline.pipeline.clustering import (
    build_base_cluster_outputs,
    fit_base_clusters,
)
from clio_pipeline.pipeline.embedding import embed_texts_in_batches
from clio_pipeline.pipeline.evaluate import (
    build_evaluation_markdown_report,
    run_synthetic_evaluation_suite,
)
from clio_pipeline.pipeline.facet_extraction import (
    extract_conversation_facets,
    extract_facets_for_conversation_batch,
)
from clio_pipeline.pipeline.hierarchy import build_multilevel_hierarchy_scaffold
from clio_pipeline.pipeline.labeling import label_clusters
from clio_pipeline.pipeline.privacy_audit import (
    apply_cluster_privacy_gate,
    audit_content,
    audit_content_batch,
    audit_facets,
    audit_labeled_clusters,
    audit_raw_conversations,
    run_privacy_auditor_validation,
    summarize_privacy_records,
)
from clio_pipeline.schemas import Conversation, Facets
from clio_pipeline.viz import (
    build_cluster_map,
    build_map_points,
    hierarchy_to_tree_view,
    project_embeddings_2d,
)

_NANOID_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def generate_run_id(size: int = 12) -> str:
    """Generate a nanoid-style run identifier."""

    return "".join(secrets.choice(_NANOID_ALPHABET) for _ in range(size))


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_run_fingerprint(
    settings: Settings,
    *,
    dataset_path: Path,
    limit: int | None,
) -> dict[str, str | int | bool | None]:
    """Build a deterministic run fingerprint for resume safety checks."""

    resolved_dataset_path = dataset_path.expanduser().resolve(strict=False)
    return {
        "dataset_path": str(resolved_dataset_path.as_posix()),
        "dataset_sha256": _sha256_file(dataset_path),
        "limit": limit,
        "openai_model": settings.resolved_openai_model(),
        "openai_base_url": settings.resolved_openai_base_url(),
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "k_base_clusters": settings.k_base_clusters,
        "clustering_strategy": settings.clustering_strategy,
        "clustering_leaf_mode": settings.clustering_leaf_mode,
        "clustering_target_leaf_size": settings.clustering_target_leaf_size,
        "clustering_min_leaf_clusters": settings.clustering_min_leaf_clusters,
        "clustering_max_leaf_clusters": settings.clustering_max_leaf_clusters,
        "clustering_hdbscan_min_cluster_size": settings.clustering_hdbscan_min_cluster_size,
        "clustering_hdbscan_min_samples": settings.clustering_hdbscan_min_samples,
        "clustering_noise_policy": settings.clustering_noise_policy,
        "random_seed": settings.random_seed,
        "facet_batch_size": settings.facet_batch_size,
        "facet_max_concurrency": settings.facet_max_concurrency,
        "cluster_label_sample_size": settings.cluster_label_sample_size,
        "cluster_label_max_concurrency": settings.cluster_label_max_concurrency,
        "hierarchy_levels": settings.hierarchy_levels,
        "hierarchy_depth_policy": settings.hierarchy_depth_policy,
        "hierarchy_target_group_size": settings.hierarchy_target_group_size,
        "hierarchy_label_max_concurrency": settings.hierarchy_label_max_concurrency,
        "privacy_threshold_min_rating": settings.privacy_threshold_min_rating,
        "privacy_batch_size": settings.privacy_batch_size,
        "privacy_max_concurrency": settings.privacy_max_concurrency,
        "privacy_validation_enabled": settings.privacy_validation_enabled,
    }


def _fingerprint_differences(
    existing: dict[str, str | int | bool | None],
    current: dict[str, str | int | bool | None],
) -> list[str]:
    """Return list of changed keys between two run fingerprints."""

    changed: list[str] = []
    for key in sorted(set(existing.keys()) | set(current.keys())):
        if existing.get(key) == current.get(key):
            continue
        changed.append(f"{key}: existing={existing.get(key)!r}, current={current.get(key)!r}")
    return changed


def _assert_resume_fingerprint_match(
    *,
    manifest: dict,
    current_fingerprint: dict[str, str | int | bool | None],
) -> None:
    """Block resume when a run fingerprint drift is detected."""

    existing_fingerprint = manifest.get("run_fingerprint")
    if not isinstance(existing_fingerprint, dict):
        return

    typed_existing = {
        str(key): value for key, value in existing_fingerprint.items() if isinstance(key, str)
    }
    differences = _fingerprint_differences(typed_existing, current_fingerprint)
    if not differences:
        return

    preview = "; ".join(differences[:5])
    if len(differences) > 5:
        preview = f"{preview}; ... {len(differences) - 5} more"
    raise ValueError(
        "Resume safety check failed: run fingerprint mismatch detected. "
        f"Differences: {preview}. "
        "Start a new run-id for changed inputs/config, or keep config/input "
        "identical when resuming."
    )


def _load_run_manifest(run_root: Path) -> dict:
    """Load an existing run manifest if present."""

    manifest_path = run_root / "run_manifest.json"
    if not manifest_path.exists():
        return {}

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return payload


def _save_run_manifest(run_root: Path, manifest: dict) -> None:
    """Persist run manifest to disk."""

    save_json(run_root / "run_manifest.json", manifest)


def _llm_metrics_snapshot(llm_client: LLMJsonClient) -> dict:
    """Read optional metrics snapshot from an LLM client."""

    snapshot_fn = getattr(llm_client, "metrics_snapshot", None)
    if not callable(snapshot_fn):
        return {}
    try:
        snapshot = snapshot_fn()
        return snapshot if isinstance(snapshot, dict) else {}
    except Exception:
        return {}


def _serialize_messages_only(conversation: Conversation) -> dict:
    """Serialize one conversation to a minimal messages-only payload."""

    return {
        "messages": [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in conversation.messages
        ]
    }


def _save_updated_conversations(
    *,
    run_root: Path,
    conversations: list[Conversation],
    facets: list[Facets] | None = None,
    assignments: list[dict] | None = None,
    cluster_labels: list[dict] | None = None,
) -> Path:
    """Save `conversation.updated.jsonl` with progressively enriched analysis fields."""

    facets_by_id = {item.conversation_id: item for item in (facets or [])}
    assignment_by_id = {str(item.get("conversation_id")): item for item in (assignments or [])}
    labels_by_cluster_id = {
        int(item["cluster_id"]): item for item in (cluster_labels or []) if "cluster_id" in item
    }

    rows: list[dict] = []
    for conversation in conversations:
        row = _serialize_messages_only(conversation)
        analysis: dict = {}

        facet = facets_by_id.get(conversation.conversation_id)
        if facet is not None:
            analysis["facets"] = facet.model_dump(mode="json")

        assignment = assignment_by_id.get(conversation.conversation_id)
        if assignment is not None:
            cluster_data = {
                "cluster_id": int(assignment["cluster_id"]),
                "kept_by_threshold": bool(assignment.get("kept_by_threshold", False)),
            }
            label = labels_by_cluster_id.get(cluster_data["cluster_id"])
            if label is not None:
                cluster_data["cluster_name"] = str(label.get("name", "")).strip()
                cluster_data["cluster_description"] = str(label.get("description", "")).strip()
                if "privacy_rating" in label:
                    cluster_data["cluster_privacy_rating"] = int(label["privacy_rating"])
                if "kept_by_privacy" in label:
                    cluster_data["kept_by_privacy"] = bool(label["kept_by_privacy"])
                if "final_kept" in label:
                    cluster_data["final_kept"] = bool(label["final_kept"])
            analysis["clustering"] = cluster_data

        if analysis:
            row["analysis"] = analysis
        rows.append(row)

    return save_jsonl(run_root / "conversation.updated.jsonl", rows)


def initialize_run_artifacts(
    *,
    settings: Settings,
    conversations: list[Conversation],
    dataset_path: Path,
    run_id: str | None = None,
    run_fingerprint: dict[str, str | int | bool | None] | None = None,
    enforce_resume_fingerprint: bool = False,
) -> tuple[str, Path]:
    """Create run folder and seed base artifacts."""

    effective_run_id = run_id or generate_run_id()
    run_root = ensure_directory(settings.output_dir / effective_run_id)
    manifest = _load_run_manifest(run_root)
    if enforce_resume_fingerprint and run_fingerprint is not None:
        _assert_resume_fingerprint_match(
            manifest=manifest,
            current_fingerprint=run_fingerprint,
        )

    conversation_path = save_jsonl(
        run_root / "conversation.jsonl",
        [_serialize_messages_only(conversation) for conversation in conversations],
    )
    updated_path = _save_updated_conversations(
        run_root=run_root,
        conversations=conversations,
    )

    completed_phases = set(manifest.get("completed_phases", []))
    completed_phases.add("phase1_dataset_load")
    output_files = dict(manifest.get("output_files", {}))
    output_files["conversation_jsonl"] = str(conversation_path.as_posix())
    output_files["conversation_updated_jsonl"] = str(updated_path.as_posix())
    manifest.update(
        {
            "run_id": effective_run_id,
            "created_at_utc": manifest.get("created_at_utc", datetime.now(UTC).isoformat()),
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "phase": "phase1_dataset_load",
            "completed_phases": sorted(completed_phases),
            "input_dataset_path": str(dataset_path.as_posix()),
            "conversation_count_input": len(conversations),
            "run_fingerprint": run_fingerprint or manifest.get("run_fingerprint", {}),
            "output_files": output_files,
        }
    )
    _save_run_manifest(run_root, manifest)
    return effective_run_id, run_root


def run_phase1_dataset_load_streaming(
    settings: Settings,
    *,
    dataset_path: Path | None = None,
    chunk_size: int = 200,
    limit: int | None = None,
) -> tuple[DatasetSummary, Path]:
    """Stream-load conversations to compute summary stats without full in-memory retention."""

    path = dataset_path or settings.input_conversations_path
    conversation_count = 0
    message_count = 0
    unique_user_ids: set[str] = set()
    min_turn_count: int | None = None
    max_turn_count = 0

    for chunk in iter_conversations_jsonl(path, chunk_size=chunk_size, limit=limit):
        for conversation in chunk:
            turns = len(conversation.messages)
            conversation_count += 1
            message_count += turns
            unique_user_ids.add(conversation.user_id)
            min_turn_count = turns if min_turn_count is None else min(min_turn_count, turns)
            max_turn_count = max(max_turn_count, turns)

    return (
        DatasetSummary(
            conversation_count=conversation_count,
            unique_user_count=len(unique_user_ids),
            message_count=message_count,
            avg_turn_count=(message_count / conversation_count) if conversation_count else 0.0,
            min_turn_count=min_turn_count if min_turn_count is not None else 0,
            max_turn_count=max_turn_count if conversation_count else 0,
        ),
        path,
    )


def initialize_run_artifacts_streaming(
    *,
    settings: Settings,
    dataset_path: Path,
    chunk_size: int,
    limit: int | None = None,
    run_id: str | None = None,
    run_fingerprint: dict[str, str | int | bool | None] | None = None,
    enforce_resume_fingerprint: bool = False,
) -> tuple[str, Path, DatasetSummary]:
    """Create run folder and seed messages-only snapshots via chunked streaming."""

    effective_run_id = run_id or generate_run_id()
    run_root = ensure_directory(settings.output_dir / effective_run_id)
    manifest = _load_run_manifest(run_root)
    if enforce_resume_fingerprint and run_fingerprint is not None:
        _assert_resume_fingerprint_match(
            manifest=manifest,
            current_fingerprint=run_fingerprint,
        )

    conversation_path = run_root / "conversation.jsonl"
    updated_path = run_root / "conversation.updated.jsonl"
    save_jsonl(conversation_path, [])
    save_jsonl(updated_path, [])

    conversation_count = 0
    message_count = 0
    unique_user_ids: set[str] = set()
    min_turn_count: int | None = None
    max_turn_count = 0

    for chunk in iter_conversations_jsonl(dataset_path, chunk_size=chunk_size, limit=limit):
        rows = [_serialize_messages_only(conversation) for conversation in chunk]
        append_jsonl(conversation_path, rows)
        append_jsonl(updated_path, rows)

        for conversation in chunk:
            turns = len(conversation.messages)
            conversation_count += 1
            message_count += turns
            unique_user_ids.add(conversation.user_id)
            min_turn_count = turns if min_turn_count is None else min(min_turn_count, turns)
            max_turn_count = max(max_turn_count, turns)

    summary = DatasetSummary(
        conversation_count=conversation_count,
        unique_user_count=len(unique_user_ids),
        message_count=message_count,
        avg_turn_count=(message_count / conversation_count) if conversation_count else 0.0,
        min_turn_count=min_turn_count if min_turn_count is not None else 0,
        max_turn_count=max_turn_count if conversation_count else 0,
    )

    completed_phases = set(manifest.get("completed_phases", []))
    completed_phases.add("phase1_dataset_load")
    output_files = dict(manifest.get("output_files", {}))
    output_files["conversation_jsonl"] = str(conversation_path.as_posix())
    output_files["conversation_updated_jsonl"] = str(updated_path.as_posix())
    manifest.update(
        {
            "run_id": effective_run_id,
            "created_at_utc": manifest.get("created_at_utc", datetime.now(UTC).isoformat()),
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "phase": "phase1_dataset_load",
            "completed_phases": sorted(completed_phases),
            "input_dataset_path": str(dataset_path.as_posix()),
            "conversation_count_input": summary.conversation_count,
            "streaming_mode": True,
            "stream_chunk_size": chunk_size,
            "run_fingerprint": run_fingerprint or manifest.get("run_fingerprint", {}),
            "output_files": output_files,
        }
    )
    _save_run_manifest(run_root, manifest)
    return effective_run_id, run_root, summary


def _load_jsonl_records(path: Path) -> list[dict]:
    """Load JSON objects from a JSONL file."""

    if not path.exists():
        return []

    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _load_json_dict(path: Path) -> dict:
    """Load one JSON object from disk, returning empty dict when missing/invalid."""

    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def load_phase2_facets(run_root: Path) -> list[Facets]:
    """Load facets output from a previous run if available."""

    facets_path = run_root / "facets" / "facets.jsonl"
    if not facets_path.exists():
        return []

    records = _load_jsonl_records(facets_path)
    return [Facets.model_validate(record) for record in records]


def load_phase3_cluster_summaries(run_root: Path) -> list[dict]:
    """Load cluster summaries from a previous run."""

    clusters_path = run_root / "clusters" / "base_clusters.json"
    if not clusters_path.exists():
        return []
    payload = json.loads(clusters_path.read_text(encoding="utf-8"))
    clusters = payload.get("clusters", [])
    return clusters if isinstance(clusters, list) else []


def load_phase4_labeled_clusters(run_root: Path) -> list[dict]:
    """Load labeled clusters from a previous run."""

    labels_path = run_root / "clusters" / "labeled_clusters.json"
    if not labels_path.exists():
        return []
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    clusters = payload.get("clusters", [])
    return clusters if isinstance(clusters, list) else []


def load_phase4_hierarchy(run_root: Path) -> dict:
    """Load hierarchy output from a previous run."""

    hierarchy_path = run_root / "clusters" / "hierarchy.json"
    if not hierarchy_path.exists():
        return {}
    payload = json.loads(hierarchy_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def load_phase5_outputs(run_root: Path) -> tuple[dict, list[dict]]:
    """Load privacy summary and gated clusters from a previous run."""

    audit_path = run_root / "privacy" / "privacy_audit.json"
    gated_path = run_root / "clusters" / "labeled_clusters_privacy_filtered.json"
    if not audit_path.exists() or not gated_path.exists():
        return {}, []

    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    gated_payload = json.loads(gated_path.read_text(encoding="utf-8"))
    summary = audit_payload.get("summary", {})
    clusters = gated_payload.get("clusters", [])
    if not isinstance(summary, dict):
        summary = {}
    if not isinstance(clusters, list):
        clusters = []
    return summary, clusters


def load_phase6_evaluation(run_root: Path) -> dict:
    """Load phase 6 evaluation metrics from a previous run."""

    path = run_root / "eval" / "phase6_metrics.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def run_phase1_dataset_load(
    settings: Settings,
    *,
    dataset_path: Path | None = None,
) -> tuple[list[Conversation], DatasetSummary, Path]:
    """Load conversations and compute summary statistics."""

    path = dataset_path or settings.input_conversations_path
    conversations = load_conversations_jsonl(path)
    summary = summarize_conversations(conversations)
    return conversations, summary, path


def run_phase2_facet_extraction(
    *,
    settings: Settings,
    conversations: list[Conversation],
    run_id: str | None = None,
    limit: int | None = None,
    llm_client: LLMJsonClient | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[list[Facets], Path]:
    """Run facet extraction and save outputs to `runs/<run_id>/facets/`."""

    if not settings.resolved_openai_api_key() and llm_client is None:
        raise ValueError(
            "OPENAI_API_KEY or AZURE_OPENAI_API_KEY is required for facet extraction. "
            "Set it in your environment or .env."
        )

    effective_run_id = run_id or generate_run_id()
    run_root = ensure_directory(settings.output_dir / effective_run_id)
    facets_dir = ensure_directory(run_root / "facets")
    conversation_path = save_jsonl(
        run_root / "conversation.jsonl",
        [_serialize_messages_only(conversation) for conversation in conversations],
    )

    selected_conversations = conversations[:limit] if limit is not None else conversations
    selected_ids = {conversation.conversation_id for conversation in selected_conversations}

    client = llm_client or OpenAIJsonClient(
        api_key=settings.resolved_openai_api_key(),
        model=settings.resolved_openai_model(),
        base_url=settings.resolved_openai_base_url(),
        temperature=settings.openai_temperature,
        max_retries=settings.client_max_retries,
        backoff_seconds=settings.client_backoff_seconds,
    )
    effective_facet_batch_size = max(1, settings.facet_batch_size)
    effective_facet_concurrency = max(1, settings.facet_max_concurrency)
    should_use_async_batch = llm_client is None and effective_facet_batch_size > 1
    phase2_execution_mode = "async_batch" if should_use_async_batch else "sync_single"
    adaptive_concurrency_enabled = bool(should_use_async_batch)

    facets_partial_path = facets_dir / "facets.partial.jsonl"
    errors_partial_path = facets_dir / "facets_errors.partial.jsonl"
    checkpoint_path = facets_dir / "facet_checkpoint.json"

    existing_partial_facets = _load_jsonl_records(facets_partial_path)
    existing_partial_errors = _load_jsonl_records(errors_partial_path)

    facets_by_id: dict[str, Facets] = {}
    for row in existing_partial_facets:
        try:
            facet = Facets.model_validate(row)
        except Exception:
            continue
        if facet.conversation_id in selected_ids:
            facets_by_id[facet.conversation_id] = facet

    extraction_errors: list[dict] = []
    for row in existing_partial_errors:
        if not isinstance(row, dict):
            continue
        conversation_id = str(row.get("conversation_id", "")).strip()
        if conversation_id and conversation_id not in selected_ids:
            continue
        extraction_errors.append(row)

    processed_ids: set[str] = set(facets_by_id.keys())
    total = len(selected_conversations)

    def _processed_count() -> int:
        return sum(
            1
            for conversation in selected_conversations
            if conversation.conversation_id in processed_ids
        )

    def _save_phase2_checkpoint(
        *,
        completed: bool,
        current_concurrency: int,
        note: str,
    ) -> None:
        save_json(
            checkpoint_path,
            {
                "phase": "phase2_facet_extraction",
                "run_id": effective_run_id,
                "execution_mode": phase2_execution_mode,
                "adaptive_concurrency_enabled": adaptive_concurrency_enabled,
                "facet_batch_size": effective_facet_batch_size,
                "facet_max_concurrency": effective_facet_concurrency,
                "current_concurrency": current_concurrency,
                "conversation_count_total": total,
                "conversation_count_processed": _processed_count(),
                "facet_count_success": len(facets_by_id),
                "error_count_recorded": len(extraction_errors),
                "completed": completed,
                "note": note,
                "updated_at_utc": datetime.now(UTC).isoformat(),
            },
        )

    resumed_count = _processed_count()
    if progress_callback is not None and resumed_count > 0:
        progress_callback(resumed_count, total, "extract_facets_resume")

    remaining_conversations = [
        conversation
        for conversation in selected_conversations
        if conversation.conversation_id not in processed_ids
    ]

    if should_use_async_batch and remaining_conversations:

        async def _extract_facets_concurrently() -> None:
            """Extract facets in adaptive concurrency waves with checkpoint writes."""

            batches = [
                remaining_conversations[index : index + effective_facet_batch_size]
                for index in range(0, len(remaining_conversations), effective_facet_batch_size)
            ]
            current_concurrency = min(
                effective_facet_concurrency,
                max(1, len(batches)),
            )
            batch_cursor = 0

            async def _extract_one_fallback(
                conversation: Conversation,
            ) -> tuple[Facets | None, dict | None]:
                try:
                    facet = await asyncio.to_thread(
                        extract_conversation_facets,
                        conversation,
                        client,
                    )
                    return facet, None
                except Exception as exc:
                    return None, {
                        "conversation_id": conversation.conversation_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "fallback_stage": "single_after_batch",
                    }

            async def _process_batch(
                batch: list[Conversation],
            ) -> tuple[list[Facets], list[dict], bool]:
                batch_facets: list[Facets] = []
                batch_errors: list[dict] = []
                had_issue = False
                try:
                    extracted, extracted_errors = await asyncio.to_thread(
                        extract_facets_for_conversation_batch,
                        batch,
                        client,
                    )
                    batch_facets.extend(extracted)
                    batch_errors.extend(extracted_errors)
                    had_issue = bool(extracted_errors)

                    extracted_ids = {facet.conversation_id for facet in extracted}
                    missing = [
                        conversation
                        for conversation in batch
                        if conversation.conversation_id not in extracted_ids
                    ]
                    for conversation in missing:
                        facet, error = await _extract_one_fallback(conversation)
                        if facet is not None:
                            batch_facets.append(facet)
                        if error is not None:
                            batch_errors.append(error)
                            had_issue = True
                except Exception as exc:
                    had_issue = True
                    batch_errors.append(
                        {
                            "conversation_id": None,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "fallback_stage": "batch_request",
                        }
                    )
                    for conversation in batch:
                        facet, error = await _extract_one_fallback(conversation)
                        if facet is not None:
                            batch_facets.append(facet)
                        if error is not None:
                            batch_errors.append(error)
                return batch_facets, batch_errors, had_issue

            while batch_cursor < len(batches):
                wave_batches = batches[batch_cursor : batch_cursor + current_concurrency]
                batch_cursor += len(wave_batches)
                tasks = [asyncio.create_task(_process_batch(batch)) for batch in wave_batches]

                wave_had_issue = False
                new_facet_rows: list[Facets] = []
                new_error_rows: list[dict] = []

                for task in tasks:
                    batch_facets, batch_errors, had_issue = await task
                    wave_had_issue = wave_had_issue or had_issue

                    for facet in batch_facets:
                        if facet.conversation_id in processed_ids:
                            continue
                        facets_by_id[facet.conversation_id] = facet
                        processed_ids.add(facet.conversation_id)
                        new_facet_rows.append(facet)

                    for error in batch_errors:
                        conversation_id = str(error.get("conversation_id", "")).strip()
                        if conversation_id and conversation_id in processed_ids:
                            continue
                        extraction_errors.append(error)
                        new_error_rows.append(error)

                if new_facet_rows:
                    append_jsonl(facets_partial_path, new_facet_rows)
                if new_error_rows:
                    append_jsonl(errors_partial_path, new_error_rows)

                if progress_callback is not None:
                    progress_callback(_processed_count(), total, "extract_facets_batch")

                if wave_had_issue and current_concurrency > 1:
                    current_concurrency -= 1
                elif (
                    not wave_had_issue
                    and current_concurrency < effective_facet_concurrency
                    and batch_cursor < len(batches)
                ):
                    current_concurrency += 1

                _save_phase2_checkpoint(
                    completed=False,
                    current_concurrency=current_concurrency,
                    note="wave_processed",
                )

        asyncio.run(_extract_facets_concurrently())
    else:
        for conversation in remaining_conversations:
            try:
                facet = extract_conversation_facets(conversation, client)
                if facet.conversation_id not in processed_ids:
                    facets_by_id[facet.conversation_id] = facet
                    processed_ids.add(facet.conversation_id)
                    append_jsonl(facets_partial_path, [facet])
            except Exception as exc:
                error_row = {
                    "conversation_id": conversation.conversation_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                extraction_errors.append(error_row)
                append_jsonl(errors_partial_path, [error_row])
            if progress_callback is not None:
                progress_callback(_processed_count(), total, "extract_facets")
            _save_phase2_checkpoint(
                completed=False,
                current_concurrency=1,
                note="single_processed",
            )

    facets = [
        facets_by_id[conversation.conversation_id]
        for conversation in selected_conversations
        if conversation.conversation_id in facets_by_id
    ]

    if not facets:
        raise ValueError("No facets were extracted successfully; aborting Phase 2.")

    extraction_errors = [
        item
        for item in extraction_errors
        if not str(item.get("conversation_id", "")).strip()
        or str(item.get("conversation_id", "")).strip() not in facets_by_id
    ]

    save_jsonl(facets_dir / "facets.jsonl", facets)
    if extraction_errors:
        save_jsonl(facets_dir / "facets_errors.jsonl", extraction_errors)

    _save_phase2_checkpoint(
        completed=True,
        current_concurrency=(effective_facet_concurrency if should_use_async_batch else 1),
        note="completed",
    )
    phase2_llm_metrics = _llm_metrics_snapshot(client)
    updated_path = _save_updated_conversations(
        run_root=run_root,
        conversations=conversations,
        facets=facets,
    )
    manifest = _load_run_manifest(run_root)
    completed_phases = set(manifest.get("completed_phases", []))
    completed_phases.add("phase2_facet_extraction")
    output_files = dict(manifest.get("output_files", {}))
    output_files["conversation_jsonl"] = str(conversation_path.as_posix())
    output_files["conversation_updated_jsonl"] = str(updated_path.as_posix())
    output_files["facets_jsonl"] = str((facets_dir / "facets.jsonl").as_posix())
    output_files["facet_checkpoint_json"] = str(checkpoint_path.as_posix())
    output_files["facets_partial_jsonl"] = str(facets_partial_path.as_posix())
    output_files["facets_errors_partial_jsonl"] = str(errors_partial_path.as_posix())
    if extraction_errors:
        output_files["facets_errors_jsonl"] = str((facets_dir / "facets_errors.jsonl").as_posix())
    manifest.update(
        {
            "run_id": effective_run_id,
            "created_at_utc": manifest.get("created_at_utc", datetime.now(UTC).isoformat()),
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "phase": "phase2_facet_extraction",
            "completed_phases": sorted(completed_phases),
            "conversation_count_input": len(conversations),
            "conversation_count_processed": len(selected_conversations),
            "facet_extraction_error_count": len(extraction_errors),
            "openai_model": settings.resolved_openai_model(),
            "openai_base_url": settings.resolved_openai_base_url(),
            "openai_key_source": settings.resolved_openai_key_source(),
            "openai_temperature": settings.openai_temperature,
            "phase2_execution_mode": phase2_execution_mode,
            "phase2_adaptive_concurrency_enabled": adaptive_concurrency_enabled,
            "facet_batch_size": effective_facet_batch_size,
            "facet_max_concurrency": effective_facet_concurrency,
            "facet_resume_processed_count": resumed_count,
            "phase2_openai_metrics": phase2_llm_metrics,
            "output_files": output_files,
        }
    )
    _save_run_manifest(run_root, manifest)

    return facets, run_root


def run_phase2_facet_extraction_streaming(
    *,
    settings: Settings,
    dataset_path: Path,
    run_id: str,
    stream_chunk_size: int,
    limit: int | None = None,
    llm_client: LLMJsonClient | None = None,
    total_conversations: int | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[list[Facets], Path]:
    """Run chunked facet extraction directly from input JSONL."""

    if not settings.resolved_openai_api_key() and llm_client is None:
        raise ValueError(
            "OPENAI_API_KEY or AZURE_OPENAI_API_KEY is required for facet extraction. "
            "Set it in your environment or .env."
        )
    if stream_chunk_size <= 0:
        raise ValueError(f"stream_chunk_size must be positive, got {stream_chunk_size}.")

    effective_run_id = run_id or generate_run_id()
    run_root = ensure_directory(settings.output_dir / effective_run_id)
    facets_dir = ensure_directory(run_root / "facets")
    conversation_path = run_root / "conversation.jsonl"
    updated_path = run_root / "conversation.updated.jsonl"

    client = llm_client or OpenAIJsonClient(
        api_key=settings.resolved_openai_api_key(),
        model=settings.resolved_openai_model(),
        base_url=settings.resolved_openai_base_url(),
        temperature=settings.openai_temperature,
        max_retries=settings.client_max_retries,
        backoff_seconds=settings.client_backoff_seconds,
    )
    effective_facet_batch_size = max(1, settings.facet_batch_size)
    effective_facet_concurrency = max(1, settings.facet_max_concurrency)
    should_use_async_batch = llm_client is None and effective_facet_batch_size > 1
    phase2_execution_mode = "async_batch" if should_use_async_batch else "sync_single"
    adaptive_concurrency_enabled = bool(should_use_async_batch)

    facets_partial_path = facets_dir / "facets.partial.jsonl"
    errors_partial_path = facets_dir / "facets_errors.partial.jsonl"
    checkpoint_path = facets_dir / "facet_checkpoint.json"

    existing_partial_facets = _load_jsonl_records(facets_partial_path)
    existing_partial_errors = _load_jsonl_records(errors_partial_path)
    facets_by_id: dict[str, Facets] = {}
    for row in existing_partial_facets:
        try:
            facet = Facets.model_validate(row)
        except Exception:
            continue
        facets_by_id[facet.conversation_id] = facet

    extraction_errors: list[dict] = []
    for row in existing_partial_errors:
        if isinstance(row, dict):
            extraction_errors.append(row)

    processed_ids: set[str] = set(facets_by_id.keys())
    total = total_conversations or (limit if limit is not None else len(processed_ids))

    def _effective_total() -> int:
        return max(1, total, len(processed_ids))

    def _save_phase2_checkpoint(
        *,
        completed: bool,
        current_concurrency: int,
        note: str,
    ) -> None:
        save_json(
            checkpoint_path,
            {
                "phase": "phase2_facet_extraction",
                "run_id": effective_run_id,
                "execution_mode": phase2_execution_mode,
                "streaming_mode": True,
                "stream_chunk_size": stream_chunk_size,
                "adaptive_concurrency_enabled": adaptive_concurrency_enabled,
                "facet_batch_size": effective_facet_batch_size,
                "facet_max_concurrency": effective_facet_concurrency,
                "current_concurrency": current_concurrency,
                "conversation_count_total": total,
                "conversation_count_processed": len(processed_ids),
                "facet_count_success": len(facets_by_id),
                "error_count_recorded": len(extraction_errors),
                "completed": completed,
                "note": note,
                "updated_at_utc": datetime.now(UTC).isoformat(),
            },
        )

    resumed_count = len(processed_ids)
    if progress_callback is not None and resumed_count > 0:
        progress_callback(resumed_count, _effective_total(), "extract_facets_resume")

    for chunk in iter_conversations_jsonl(
        dataset_path,
        chunk_size=stream_chunk_size,
        limit=limit,
    ):
        remaining_conversations = [
            conversation
            for conversation in chunk
            if conversation.conversation_id not in processed_ids
        ]
        if not remaining_conversations:
            continue
        chunk_conversations = list(remaining_conversations)

        if should_use_async_batch:

            async def _extract_chunk_concurrently(
                chunk_items: list[Conversation],
            ) -> None:
                """Extract one chunk with adaptive waves."""

                batches = [
                    chunk_items[index : index + effective_facet_batch_size]
                    for index in range(
                        0,
                        len(chunk_items),
                        effective_facet_batch_size,
                    )
                ]
                current_concurrency = min(
                    effective_facet_concurrency,
                    max(1, len(batches)),
                )
                batch_cursor = 0

                async def _extract_one_fallback(
                    conversation: Conversation,
                ) -> tuple[Facets | None, dict | None]:
                    try:
                        facet = await asyncio.to_thread(
                            extract_conversation_facets,
                            conversation,
                            client,
                        )
                        return facet, None
                    except Exception as exc:
                        return None, {
                            "conversation_id": conversation.conversation_id,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "fallback_stage": "single_after_batch",
                        }

                async def _process_batch(
                    batch: list[Conversation],
                ) -> tuple[list[Facets], list[dict], bool]:
                    batch_facets: list[Facets] = []
                    batch_errors: list[dict] = []
                    had_issue = False
                    try:
                        extracted, extracted_errors = await asyncio.to_thread(
                            extract_facets_for_conversation_batch,
                            batch,
                            client,
                        )
                        batch_facets.extend(extracted)
                        batch_errors.extend(extracted_errors)
                        had_issue = bool(extracted_errors)

                        extracted_ids = {facet.conversation_id for facet in extracted}
                        missing = [
                            conversation
                            for conversation in batch
                            if conversation.conversation_id not in extracted_ids
                        ]
                        for conversation in missing:
                            facet, error = await _extract_one_fallback(conversation)
                            if facet is not None:
                                batch_facets.append(facet)
                            if error is not None:
                                batch_errors.append(error)
                                had_issue = True
                    except Exception as exc:
                        had_issue = True
                        batch_errors.append(
                            {
                                "conversation_id": None,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                                "fallback_stage": "batch_request",
                            }
                        )
                        for conversation in batch:
                            facet, error = await _extract_one_fallback(conversation)
                            if facet is not None:
                                batch_facets.append(facet)
                            if error is not None:
                                batch_errors.append(error)
                    return batch_facets, batch_errors, had_issue

                while batch_cursor < len(batches):
                    wave_batches = batches[batch_cursor : batch_cursor + current_concurrency]
                    batch_cursor += len(wave_batches)
                    tasks = [asyncio.create_task(_process_batch(batch)) for batch in wave_batches]

                    wave_had_issue = False
                    new_facet_rows: list[Facets] = []
                    new_error_rows: list[dict] = []
                    for task in tasks:
                        batch_facets, batch_errors, had_issue = await task
                        wave_had_issue = wave_had_issue or had_issue
                        for facet in batch_facets:
                            if facet.conversation_id in processed_ids:
                                continue
                            facets_by_id[facet.conversation_id] = facet
                            processed_ids.add(facet.conversation_id)
                            new_facet_rows.append(facet)

                        for error in batch_errors:
                            conversation_id = str(error.get("conversation_id", "")).strip()
                            if conversation_id and conversation_id in processed_ids:
                                continue
                            extraction_errors.append(error)
                            new_error_rows.append(error)

                    if new_facet_rows:
                        append_jsonl(facets_partial_path, new_facet_rows)
                    if new_error_rows:
                        append_jsonl(errors_partial_path, new_error_rows)

                    if progress_callback is not None:
                        progress_callback(
                            len(processed_ids),
                            _effective_total(),
                            "extract_facets_batch",
                        )

                    if wave_had_issue and current_concurrency > 1:
                        current_concurrency -= 1
                    elif (
                        not wave_had_issue
                        and current_concurrency < effective_facet_concurrency
                        and batch_cursor < len(batches)
                    ):
                        current_concurrency += 1

                    _save_phase2_checkpoint(
                        completed=False,
                        current_concurrency=current_concurrency,
                        note="wave_processed",
                    )

            asyncio.run(_extract_chunk_concurrently(chunk_conversations))
        else:
            for conversation in chunk_conversations:
                try:
                    facet = extract_conversation_facets(conversation, client)
                    if facet.conversation_id not in processed_ids:
                        facets_by_id[facet.conversation_id] = facet
                        processed_ids.add(facet.conversation_id)
                        append_jsonl(facets_partial_path, [facet])
                except Exception as exc:
                    error_row = {
                        "conversation_id": conversation.conversation_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    extraction_errors.append(error_row)
                    append_jsonl(errors_partial_path, [error_row])

                if progress_callback is not None:
                    progress_callback(
                        len(processed_ids),
                        _effective_total(),
                        "extract_facets",
                    )
                _save_phase2_checkpoint(
                    completed=False,
                    current_concurrency=1,
                    note="single_processed",
                )

    if not facets_by_id:
        raise ValueError("No facets were extracted successfully; aborting Phase 2.")

    extraction_errors = [
        item
        for item in extraction_errors
        if not str(item.get("conversation_id", "")).strip()
        or str(item.get("conversation_id", "")).strip() not in facets_by_id
    ]

    ordered_facets: list[Facets] = []
    for chunk in iter_conversations_jsonl(
        dataset_path,
        chunk_size=stream_chunk_size,
        limit=limit,
    ):
        for conversation in chunk:
            facet = facets_by_id.get(conversation.conversation_id)
            if facet is not None:
                ordered_facets.append(facet)

    save_jsonl(updated_path, [])
    for chunk in iter_conversations_jsonl(
        dataset_path,
        chunk_size=stream_chunk_size,
        limit=limit,
    ):
        updated_rows: list[dict] = []
        for conversation in chunk:
            row = _serialize_messages_only(conversation)
            facet = facets_by_id.get(conversation.conversation_id)
            if facet is not None:
                row["analysis"] = {"facets": facet.model_dump(mode="json")}
            updated_rows.append(row)
        append_jsonl(updated_path, updated_rows)

    save_jsonl(facets_dir / "facets.jsonl", ordered_facets)
    if extraction_errors:
        save_jsonl(facets_dir / "facets_errors.jsonl", extraction_errors)

    _save_phase2_checkpoint(
        completed=True,
        current_concurrency=(effective_facet_concurrency if should_use_async_batch else 1),
        note="completed",
    )
    phase2_llm_metrics = _llm_metrics_snapshot(client)

    manifest = _load_run_manifest(run_root)
    completed_phases = set(manifest.get("completed_phases", []))
    completed_phases.add("phase2_facet_extraction")
    output_files = dict(manifest.get("output_files", {}))
    output_files["conversation_jsonl"] = str(conversation_path.as_posix())
    output_files["conversation_updated_jsonl"] = str(updated_path.as_posix())
    output_files["facets_jsonl"] = str((facets_dir / "facets.jsonl").as_posix())
    output_files["facet_checkpoint_json"] = str(checkpoint_path.as_posix())
    output_files["facets_partial_jsonl"] = str(facets_partial_path.as_posix())
    output_files["facets_errors_partial_jsonl"] = str(errors_partial_path.as_posix())
    if extraction_errors:
        output_files["facets_errors_jsonl"] = str((facets_dir / "facets_errors.jsonl").as_posix())

    manifest.update(
        {
            "run_id": effective_run_id,
            "created_at_utc": manifest.get("created_at_utc", datetime.now(UTC).isoformat()),
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "phase": "phase2_facet_extraction",
            "completed_phases": sorted(completed_phases),
            "conversation_count_input": int(manifest.get("conversation_count_input", 0)),
            "conversation_count_processed": len(ordered_facets),
            "facet_extraction_error_count": len(extraction_errors),
            "openai_model": settings.resolved_openai_model(),
            "openai_base_url": settings.resolved_openai_base_url(),
            "openai_key_source": settings.resolved_openai_key_source(),
            "openai_temperature": settings.openai_temperature,
            "phase2_execution_mode": phase2_execution_mode,
            "phase2_streaming_mode": True,
            "stream_chunk_size": stream_chunk_size,
            "phase2_adaptive_concurrency_enabled": adaptive_concurrency_enabled,
            "facet_batch_size": effective_facet_batch_size,
            "facet_max_concurrency": effective_facet_concurrency,
            "facet_resume_processed_count": resumed_count,
            "phase2_openai_metrics": phase2_llm_metrics,
            "output_files": output_files,
        }
    )
    _save_run_manifest(run_root, manifest)
    return ordered_facets, run_root


def run_phase3_base_clustering(
    *,
    settings: Settings,
    conversations: list[Conversation],
    facets: list[Facets],
    run_root: Path,
    embedding_client: TextEmbeddingClient | None = None,
    batch_size: int | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[list[dict], Path]:
    """Run embeddings + base k-means clustering and save artifacts."""

    if not facets:
        raise ValueError("No facets provided for clustering.")
    if not settings.jina_api_key and embedding_client is None:
        raise ValueError(
            "JINA_API_KEY is required for embedding extraction. Set it in your environment or .env."
        )
    if settings.embedding_provider.lower() != "jina" and embedding_client is None:
        raise ValueError(
            f"Unsupported embedding_provider '{settings.embedding_provider}'. "
            "Only 'jina' is implemented."
        )

    effective_batch_size = batch_size or settings.embedding_batch_size
    texts = [facet.summary for facet in facets]

    created_client: JinaEmbeddingClient | None = None
    client: TextEmbeddingClient
    if embedding_client is not None:
        client = embedding_client
    else:
        created_client = JinaEmbeddingClient(
            api_key=settings.jina_api_key,
            model=settings.embedding_model,
            max_retries=settings.client_max_retries,
            backoff_seconds=settings.client_backoff_seconds,
        )
        client = created_client

    try:
        embeddings = embed_texts_in_batches(
            texts,
            client,
            batch_size=effective_batch_size,
            progress_callback=(
                (lambda done, total: progress_callback(done, total, "embed_summaries"))
                if progress_callback is not None
                else None
            ),
        )
    finally:
        if created_client is not None:
            created_client.close()

    clustering_result = fit_base_clusters(
        embeddings,
        strategy=settings.clustering_strategy,
        leaf_mode=settings.clustering_leaf_mode,
        requested_k=settings.k_base_clusters,
        target_leaf_size=settings.clustering_target_leaf_size,
        min_leaf_clusters=settings.clustering_min_leaf_clusters,
        max_leaf_clusters=settings.clustering_max_leaf_clusters,
        hdbscan_min_cluster_size=settings.clustering_hdbscan_min_cluster_size,
        hdbscan_min_samples=settings.clustering_hdbscan_min_samples,
        noise_policy=settings.clustering_noise_policy,
        random_seed=settings.random_seed,
    )
    labels = clustering_result.labels
    centroids = clustering_result.centroids
    effective_k = clustering_result.effective_k
    cluster_summaries, assignments = build_base_cluster_outputs(
        conversations=conversations,
        facets=facets,
        labels=labels,
        min_unique_users=settings.min_unique_users,
        min_conversations_per_cluster=settings.min_conversations_per_cluster,
    )

    embeddings_dir = ensure_directory(run_root / "embeddings")
    clusters_dir = ensure_directory(run_root / "clusters")
    viz_dir = ensure_directory(run_root / "viz")

    np.save(embeddings_dir / "summary_embeddings.npy", embeddings)
    np.save(clusters_dir / "base_centroids.npy", centroids)
    save_jsonl(clusters_dir / "base_assignments.jsonl", assignments)
    projection_coords, projection_method = project_embeddings_2d(
        embeddings,
        method=settings.viz_projection_method,
        random_seed=settings.random_seed,
    )
    map_points = build_map_points(
        facets=facets,
        assignments=assignments,
        coords=projection_coords,
    )
    map_clusters = build_cluster_map(map_points)
    save_jsonl(viz_dir / "map_points.jsonl", map_points)
    save_json(
        viz_dir / "map_clusters.json",
        {
            "projection_method_requested": settings.viz_projection_method,
            "projection_method_used": projection_method,
            "cluster_count": len(map_clusters),
            "clusters": map_clusters,
        },
    )
    updated_path = _save_updated_conversations(
        run_root=run_root,
        conversations=conversations,
        facets=facets,
        assignments=assignments,
    )
    save_json(
        clusters_dir / "base_clusters.json",
        {
            "requested_k": settings.k_base_clusters,
            "resolved_requested_k": clustering_result.requested_k,
            "effective_k": effective_k,
            "clustering_strategy": clustering_result.strategy,
            "clustering_leaf_mode": clustering_result.leaf_mode,
            "clustering_noise_policy": clustering_result.noise_policy,
            "clustering_auto_target_k": clustering_result.auto_target_k,
            "clustering_raw_cluster_count": clustering_result.raw_cluster_count,
            "clustering_noise_count": clustering_result.noise_count,
            "clustering_noise_rate": (clustering_result.noise_count / max(1, embeddings.shape[0])),
            "clustering_refinement_splits": clustering_result.refinement_splits,
            "clustering_silhouette_score": clustering_result.silhouette_score,
            "clustering_davies_bouldin_score": clustering_result.davies_bouldin_score,
            "clustering_calinski_harabasz_score": clustering_result.calinski_harabasz_score,
            "clustering_fallback_reason": clustering_result.fallback_reason,
            "cluster_count_total": len(cluster_summaries),
            "cluster_count_kept": sum(1 for item in cluster_summaries if item["kept_by_threshold"]),
            "clusters": cluster_summaries,
        },
    )

    manifest = _load_run_manifest(run_root)
    completed_phases = set(manifest.get("completed_phases", []))
    completed_phases.add("phase3_base_clustering")
    output_files = dict(manifest.get("output_files", {}))
    output_files.update(
        {
            "conversation_updated_jsonl": str(updated_path.as_posix()),
            "summary_embeddings_npy": str((embeddings_dir / "summary_embeddings.npy").as_posix()),
            "base_centroids_npy": str((clusters_dir / "base_centroids.npy").as_posix()),
            "base_assignments_jsonl": str((clusters_dir / "base_assignments.jsonl").as_posix()),
            "base_clusters_json": str((clusters_dir / "base_clusters.json").as_posix()),
            "viz_map_points_jsonl": str((viz_dir / "map_points.jsonl").as_posix()),
            "viz_map_clusters_json": str((viz_dir / "map_clusters.json").as_posix()),
        }
    )
    manifest.update(
        {
            "run_id": manifest.get("run_id", run_root.name),
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "phase": "phase3_base_clustering",
            "completed_phases": sorted(completed_phases),
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
            "embedding_batch_size": effective_batch_size,
            "clustering_strategy": clustering_result.strategy,
            "clustering_leaf_mode": clustering_result.leaf_mode,
            "clustering_noise_policy": clustering_result.noise_policy,
            "clustering_auto_target_k": clustering_result.auto_target_k,
            "clustering_target_leaf_size": settings.clustering_target_leaf_size,
            "clustering_min_leaf_clusters": settings.clustering_min_leaf_clusters,
            "clustering_max_leaf_clusters": settings.clustering_max_leaf_clusters,
            "clustering_hdbscan_min_cluster_size": settings.clustering_hdbscan_min_cluster_size,
            "clustering_hdbscan_min_samples": settings.clustering_hdbscan_min_samples,
            "requested_k": settings.k_base_clusters,
            "resolved_requested_k": clustering_result.requested_k,
            "effective_k": effective_k,
            "clustering_raw_cluster_count": clustering_result.raw_cluster_count,
            "clustering_noise_count": clustering_result.noise_count,
            "clustering_refinement_splits": clustering_result.refinement_splits,
            "clustering_silhouette_score": clustering_result.silhouette_score,
            "clustering_davies_bouldin_score": clustering_result.davies_bouldin_score,
            "clustering_calinski_harabasz_score": clustering_result.calinski_harabasz_score,
            "clustering_fallback_reason": clustering_result.fallback_reason,
            "viz_projection_method": projection_method,
            "cluster_count_total": len(cluster_summaries),
            "cluster_count_kept": sum(1 for item in cluster_summaries if item["kept_by_threshold"]),
            "output_files": output_files,
        }
    )
    _save_run_manifest(run_root, manifest)

    return cluster_summaries, run_root


def run_phase4_cluster_labeling(
    *,
    settings: Settings,
    facets: list[Facets],
    cluster_summaries: list[dict],
    run_root: Path,
    conversations: list[Conversation] | None = None,
    llm_client: LLMJsonClient | None = None,
    sample_size: int | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[list[dict], Path]:
    """Run cluster labeling and save labeled cluster metadata."""

    if not settings.resolved_openai_api_key() and llm_client is None:
        raise ValueError(
            "OPENAI_API_KEY or AZURE_OPENAI_API_KEY is required for cluster labeling. "
            "Set it in your environment or .env."
        )
    if not cluster_summaries:
        raise ValueError("No cluster summaries provided for labeling.")

    client = llm_client or OpenAIJsonClient(
        api_key=settings.resolved_openai_api_key(),
        model=settings.resolved_openai_model(),
        base_url=settings.resolved_openai_base_url(),
        temperature=settings.openai_temperature,
        max_retries=settings.client_max_retries,
        backoff_seconds=settings.client_backoff_seconds,
    )
    effective_sample_size = sample_size or settings.cluster_label_sample_size
    configured_label_concurrency = max(1, settings.cluster_label_max_concurrency)
    should_use_parallel = llm_client is None and configured_label_concurrency > 1
    label_max_concurrency = configured_label_concurrency if should_use_parallel else 1
    phase4_execution_mode = "parallel" if should_use_parallel else "sync"
    clusters_dir = ensure_directory(run_root / "clusters")
    adaptive_concurrency_enabled = bool(should_use_parallel)
    labels_partial_path = clusters_dir / "labeled_clusters.partial.jsonl"
    errors_partial_path = clusters_dir / "labeled_clusters_errors.partial.jsonl"
    checkpoint_path = clusters_dir / "cluster_label_checkpoint.json"

    ordered_cluster_summaries = sorted(
        cluster_summaries,
        key=lambda item: int(item["cluster_id"]),
    )
    expected_cluster_ids = {int(item["cluster_id"]) for item in ordered_cluster_summaries}
    labeled_by_id: dict[int, dict] = {}
    for row in _load_jsonl_records(labels_partial_path):
        if not isinstance(row, dict):
            continue
        cluster_id_raw = row.get("cluster_id")
        try:
            cluster_id = int(cluster_id_raw)
        except (TypeError, ValueError):
            continue
        if cluster_id in expected_cluster_ids:
            labeled_by_id[cluster_id] = row

    processed_cluster_ids: set[int] = set(labeled_by_id.keys())
    total = len(ordered_cluster_summaries)

    def _processed_count() -> int:
        return sum(
            1
            for cluster in ordered_cluster_summaries
            if int(cluster["cluster_id"]) in processed_cluster_ids
        )

    def _save_cluster_label_checkpoint(
        *,
        completed: bool,
        current_concurrency: int,
        note: str,
    ) -> None:
        save_json(
            checkpoint_path,
            {
                "phase": "phase4_cluster_labeling",
                "run_id": run_root.name,
                "execution_mode": phase4_execution_mode,
                "adaptive_concurrency_enabled": adaptive_concurrency_enabled,
                "cluster_label_sample_size": effective_sample_size,
                "cluster_label_max_concurrency": label_max_concurrency,
                "current_concurrency": current_concurrency,
                "cluster_total": total,
                "cluster_processed": _processed_count(),
                "cluster_labeled_count": len(labeled_by_id),
                "cluster_fallback_count": sum(
                    1
                    for item in labeled_by_id.values()
                    if bool(item.get("labeling_fallback_used", False))
                ),
                "completed": completed,
                "note": note,
                "updated_at_utc": datetime.now(UTC).isoformat(),
            },
        )

    resumed_count = _processed_count()
    if progress_callback is not None and resumed_count > 0:
        progress_callback(resumed_count, total, "label_clusters_resume")

    remaining_clusters = [
        cluster
        for cluster in ordered_cluster_summaries
        if int(cluster["cluster_id"]) not in processed_cluster_ids
    ]

    if should_use_parallel and remaining_clusters:
        cluster_cursor = 0
        current_concurrency = min(label_max_concurrency, max(1, len(remaining_clusters)))
        while cluster_cursor < len(remaining_clusters):
            wave_clusters = remaining_clusters[
                cluster_cursor : cluster_cursor + current_concurrency
            ]
            cluster_cursor += len(wave_clusters)
            wave_results = label_clusters(
                cluster_summaries=wave_clusters,
                facets=facets,
                llm_client=client,
                sample_size=effective_sample_size,
                max_concurrency=current_concurrency,
            )

            wave_had_issue = False
            new_rows: list[dict] = []
            for row in wave_results:
                cluster_id = int(row["cluster_id"])
                if cluster_id in processed_cluster_ids:
                    continue
                processed_cluster_ids.add(cluster_id)
                labeled_by_id[cluster_id] = row
                new_rows.append(row)
                if bool(row.get("labeling_fallback_used", False)):
                    wave_had_issue = True

            if new_rows:
                append_jsonl(labels_partial_path, new_rows)
                new_error_rows = [
                    {
                        "cluster_id": int(item["cluster_id"]),
                        "error": str(item.get("labeling_error", "")),
                    }
                    for item in new_rows
                    if bool(item.get("labeling_fallback_used", False))
                ]
                if new_error_rows:
                    append_jsonl(errors_partial_path, new_error_rows)

            if progress_callback is not None:
                progress_callback(_processed_count(), total, "label_clusters")

            if wave_had_issue and current_concurrency > 1:
                current_concurrency -= 1
            elif (
                not wave_had_issue
                and current_concurrency < label_max_concurrency
                and cluster_cursor < len(remaining_clusters)
            ):
                current_concurrency += 1

            _save_cluster_label_checkpoint(
                completed=False,
                current_concurrency=current_concurrency,
                note="wave_processed",
            )
    else:
        for cluster in remaining_clusters:
            result = label_clusters(
                cluster_summaries=[cluster],
                facets=facets,
                llm_client=client,
                sample_size=effective_sample_size,
                max_concurrency=1,
            )[0]
            cluster_id = int(result["cluster_id"])
            if cluster_id not in processed_cluster_ids:
                processed_cluster_ids.add(cluster_id)
                labeled_by_id[cluster_id] = result
                append_jsonl(labels_partial_path, [result])
                if bool(result.get("labeling_fallback_used", False)):
                    append_jsonl(
                        errors_partial_path,
                        [
                            {
                                "cluster_id": cluster_id,
                                "error": str(result.get("labeling_error", "")),
                            }
                        ],
                    )

            if progress_callback is not None:
                progress_callback(_processed_count(), total, "label_clusters")

            _save_cluster_label_checkpoint(
                completed=False,
                current_concurrency=1,
                note="single_processed",
            )

    labeled_clusters = [
        labeled_by_id[int(cluster["cluster_id"])]
        for cluster in ordered_cluster_summaries
        if int(cluster["cluster_id"]) in labeled_by_id
    ]
    if not labeled_clusters:
        raise ValueError("No clusters were labeled successfully; aborting Phase 4 labeling.")

    labeling_errors = [
        {
            "cluster_id": int(item["cluster_id"]),
            "error": str(item.get("labeling_error", "")),
        }
        for item in labeled_clusters
        if bool(item.get("labeling_fallback_used", False))
    ]
    phase4_label_llm_metrics = _llm_metrics_snapshot(client)

    save_json(
        clusters_dir / "labeled_clusters.json",
        {
            "cluster_count_total": len(labeled_clusters),
            "cluster_count_kept": sum(1 for item in labeled_clusters if item["kept_by_threshold"]),
            "labeling_fallback_count": len(labeling_errors),
            "clusters": labeled_clusters,
        },
    )
    if labeling_errors:
        save_jsonl(clusters_dir / "labeled_clusters_errors.jsonl", labeling_errors)

    _save_cluster_label_checkpoint(
        completed=True,
        current_concurrency=(label_max_concurrency if should_use_parallel else 1),
        note="completed",
    )

    updated_path: Path | None = None
    assignments_path = clusters_dir / "base_assignments.jsonl"
    assignment_records = _load_jsonl_records(assignments_path)
    if assignment_records and conversations:
        updated_path = _save_updated_conversations(
            run_root=run_root,
            conversations=conversations,
            facets=facets,
            assignments=assignment_records,
            cluster_labels=labeled_clusters,
        )

    manifest = _load_run_manifest(run_root)
    completed_phases = set(manifest.get("completed_phases", []))
    completed_phases.add("phase4_cluster_labeling")
    output_files = dict(manifest.get("output_files", {}))
    if updated_path is not None:
        output_files["conversation_updated_jsonl"] = str(updated_path.as_posix())
    output_files["labeled_clusters_json"] = str((clusters_dir / "labeled_clusters.json").as_posix())
    output_files["cluster_label_checkpoint_json"] = str(checkpoint_path.as_posix())
    output_files["labeled_clusters_partial_jsonl"] = str(labels_partial_path.as_posix())
    output_files["labeled_clusters_errors_partial_jsonl"] = str(errors_partial_path.as_posix())
    if labeling_errors:
        output_files["labeled_clusters_errors_jsonl"] = str(
            (clusters_dir / "labeled_clusters_errors.jsonl").as_posix()
        )
    manifest.update(
        {
            "run_id": manifest.get("run_id", run_root.name),
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "phase": "phase4_cluster_labeling",
            "completed_phases": sorted(completed_phases),
            "cluster_label_sample_size": effective_sample_size,
            "cluster_count_total": len(labeled_clusters),
            "cluster_count_kept": sum(1 for item in labeled_clusters if item["kept_by_threshold"]),
            "cluster_label_fallback_count": len(labeling_errors),
            "phase4_execution_mode": phase4_execution_mode,
            "phase4_cluster_adaptive_concurrency_enabled": adaptive_concurrency_enabled,
            "cluster_label_max_concurrency": label_max_concurrency,
            "cluster_label_resume_processed_count": resumed_count,
            "phase4_label_openai_metrics": phase4_label_llm_metrics,
            "output_files": output_files,
        }
    )
    _save_run_manifest(run_root, manifest)

    return labeled_clusters, run_root


def run_phase4_hierarchy_scaffold(
    *,
    settings: Settings,
    labeled_clusters: list[dict],
    run_root: Path,
    llm_client: LLMJsonClient | None = None,
    embedding_client: TextEmbeddingClient | None = None,
    batch_size: int | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[dict, Path]:
    """Build and save a multi-level hierarchy over labeled clusters."""

    if not labeled_clusters:
        raise ValueError("No labeled clusters available for hierarchy construction.")
    if not settings.resolved_openai_api_key() and llm_client is None:
        raise ValueError(
            "OPENAI_API_KEY or AZURE_OPENAI_API_KEY is required for hierarchy labeling. "
            "Set it in your environment or .env."
        )
    if not settings.jina_api_key and embedding_client is None:
        raise ValueError(
            "JINA_API_KEY is required for hierarchy embeddings. Set it in your environment or .env."
        )
    if settings.embedding_provider.lower() != "jina" and embedding_client is None:
        raise ValueError(
            f"Unsupported embedding_provider '{settings.embedding_provider}'. "
            "Only 'jina' is implemented."
        )

    llm = llm_client or OpenAIJsonClient(
        api_key=settings.resolved_openai_api_key(),
        model=settings.resolved_openai_model(),
        base_url=settings.resolved_openai_base_url(),
        temperature=settings.openai_temperature,
        max_retries=settings.client_max_retries,
        backoff_seconds=settings.client_backoff_seconds,
    )
    configured_hierarchy_concurrency = max(1, settings.hierarchy_label_max_concurrency)
    should_use_parallel = llm_client is None and configured_hierarchy_concurrency > 1
    hierarchy_label_max_concurrency = configured_hierarchy_concurrency if should_use_parallel else 1
    phase4_hierarchy_execution_mode = "parallel" if should_use_parallel else "sync"
    adaptive_concurrency_enabled = bool(should_use_parallel)
    effective_batch_size = batch_size or settings.embedding_batch_size
    clusters_dir = ensure_directory(run_root / "clusters")
    hierarchy_groups_partial_path = clusters_dir / "hierarchy_label_groups.partial.jsonl"
    hierarchy_checkpoint_path = clusters_dir / "hierarchy_checkpoint.json"

    existing_label_results: dict[str, dict] = {}
    for row in _load_jsonl_records(hierarchy_groups_partial_path):
        if not isinstance(row, dict):
            continue
        key = str(row.get("key", "")).strip()
        name = str(row.get("name", "")).strip()
        description = str(row.get("description", "")).strip()
        if not key or not name or not description:
            continue
        existing_label_results[key] = {
            "name": name,
            "description": description,
            "fallback_used": bool(row.get("fallback_used", False)),
            "fallback_error": row.get("fallback_error"),
            "fallback_record": row.get("fallback_record"),
        }

    def _save_hierarchy_checkpoint(
        *,
        completed: bool,
        current_concurrency: int,
        note: str,
    ) -> None:
        save_json(
            hierarchy_checkpoint_path,
            {
                "phase": "phase4_hierarchy_scaffold",
                "run_id": run_root.name,
                "execution_mode": phase4_hierarchy_execution_mode,
                "adaptive_concurrency_enabled": adaptive_concurrency_enabled,
                "hierarchy_levels": settings.hierarchy_levels,
                "hierarchy_depth_policy": settings.hierarchy_depth_policy,
                "hierarchy_target_group_size": settings.hierarchy_target_group_size,
                "hierarchy_label_max_concurrency": hierarchy_label_max_concurrency,
                "current_concurrency": current_concurrency,
                "label_checkpoint_count": len(existing_label_results),
                "completed": completed,
                "note": note,
                "updated_at_utc": datetime.now(UTC).isoformat(),
            },
        )

    def _on_hierarchy_label_result(record: dict) -> None:
        key = str(record.get("key", "")).strip()
        name = str(record.get("name", "")).strip()
        description = str(record.get("description", "")).strip()
        if not key or not name or not description:
            return
        if key in existing_label_results:
            return

        existing_label_results[key] = {
            "name": name,
            "description": description,
            "fallback_used": bool(record.get("fallback_used", False)),
            "fallback_error": record.get("fallback_error"),
            "fallback_record": record.get("fallback_record"),
        }
        append_jsonl(
            hierarchy_groups_partial_path,
            [
                {
                    "key": key,
                    "level": int(record.get("level", 0)),
                    "group_offset": int(record.get("group_offset", 0)),
                    "group_id": int(record.get("group_id", 0)),
                    "child_count": int(record.get("child_count", 0)),
                    "level_concurrency": int(record.get("level_concurrency", 1)),
                    "name": name,
                    "description": description,
                    "fallback_used": bool(record.get("fallback_used", False)),
                    "fallback_error": record.get("fallback_error"),
                    "fallback_record": record.get("fallback_record"),
                }
            ],
        )
        _save_hierarchy_checkpoint(
            completed=False,
            current_concurrency=int(record.get("level_concurrency", 1)),
            note="label_group_processed",
        )

    texts = [f"{item['name']}. {item['description']}" for item in labeled_clusters]

    created_client: JinaEmbeddingClient | None = None
    client: TextEmbeddingClient
    if embedding_client is not None:
        client = embedding_client
    else:
        created_client = JinaEmbeddingClient(
            api_key=settings.jina_api_key,
            model=settings.embedding_model,
            max_retries=settings.client_max_retries,
            backoff_seconds=settings.client_backoff_seconds,
        )
        client = created_client

    try:
        embeddings = embed_texts_in_batches(
            texts,
            client,
            batch_size=effective_batch_size,
            progress_callback=(
                (lambda done, total: progress_callback(done, total, "embed_cluster_labels"))
                if progress_callback is not None
                else None
            ),
        )
    finally:
        if created_client is not None:
            created_client.close()

    _save_hierarchy_checkpoint(
        completed=False,
        current_concurrency=(hierarchy_label_max_concurrency if should_use_parallel else 1),
        note="embeddings_ready",
    )

    hierarchy = build_multilevel_hierarchy_scaffold(
        labeled_clusters=labeled_clusters,
        leaf_embeddings=embeddings,
        llm_client=llm,
        requested_levels=settings.hierarchy_levels,
        target_group_size=settings.hierarchy_target_group_size,
        random_seed=settings.random_seed,
        depth_policy=settings.hierarchy_depth_policy,
        max_label_concurrency=hierarchy_label_max_concurrency,
        progress_callback=(
            (lambda done, total: progress_callback(done, total, "label_hierarchy_groups"))
            if progress_callback is not None
            else None
        ),
        adaptive_concurrency=adaptive_concurrency_enabled,
        existing_label_results=existing_label_results,
        checkpoint_callback=_on_hierarchy_label_result,
    )
    hierarchy["requested_top_k"] = settings.hierarchy_top_k
    phase4_hierarchy_llm_metrics = _llm_metrics_snapshot(llm)

    _save_hierarchy_checkpoint(
        completed=True,
        current_concurrency=int(
            hierarchy.get(
                "hierarchy_label_final_concurrency",
                hierarchy_label_max_concurrency if should_use_parallel else 1,
            )
        ),
        note="completed",
    )

    viz_dir = ensure_directory(run_root / "viz")
    hierarchy_build_report = {
        "run_id": run_root.name,
        "requested_levels": int(hierarchy.get("requested_levels", settings.hierarchy_levels)),
        "generated_levels": int(hierarchy.get("generated_levels", 1)),
        "depth_policy": str(hierarchy.get("depth_policy", settings.hierarchy_depth_policy)),
        "depth_stop_reason": hierarchy.get("depth_stop_reason"),
        "depth_stop_details": hierarchy.get("depth_stop_details"),
        "why_not_deeper": hierarchy.get("why_not_deeper"),
        "level_build_stats": hierarchy.get("level_build_stats", []),
        "top_level_cluster_count": int(hierarchy.get("top_level_cluster_count", 0)),
        "leaf_cluster_count": int(hierarchy.get("leaf_cluster_count", 0)),
        "updated_at_utc": datetime.now(UTC).isoformat(),
    }
    save_json(clusters_dir / "hierarchy.json", hierarchy)
    save_json(clusters_dir / "hierarchy_build_report.json", hierarchy_build_report)
    save_json(viz_dir / "tree_view.json", hierarchy_to_tree_view(hierarchy))
    if hierarchy.get("hierarchy_label_fallback_records"):
        save_json(
            clusters_dir / "hierarchy_label_errors.json",
            {
                "count": int(hierarchy.get("hierarchy_label_fallback_count", 0)),
                "records": hierarchy.get("hierarchy_label_fallback_records", []),
            },
        )

    manifest = _load_run_manifest(run_root)
    completed_phases = set(manifest.get("completed_phases", []))
    completed_phases.add("phase4_hierarchy_scaffold")
    output_files = dict(manifest.get("output_files", {}))
    output_files["hierarchy_json"] = str((clusters_dir / "hierarchy.json").as_posix())
    output_files["hierarchy_build_report_json"] = str(
        (clusters_dir / "hierarchy_build_report.json").as_posix()
    )
    output_files["tree_view_json"] = str((viz_dir / "tree_view.json").as_posix())
    output_files["hierarchy_checkpoint_json"] = str(hierarchy_checkpoint_path.as_posix())
    output_files["hierarchy_label_groups_partial_jsonl"] = str(
        hierarchy_groups_partial_path.as_posix()
    )
    if hierarchy.get("hierarchy_label_fallback_records"):
        output_files["hierarchy_label_errors_json"] = str(
            (clusters_dir / "hierarchy_label_errors.json").as_posix()
        )
    manifest.update(
        {
            "run_id": manifest.get("run_id", run_root.name),
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "phase": "phase4_hierarchy_scaffold",
            "completed_phases": sorted(completed_phases),
            "requested_top_k": settings.hierarchy_top_k,
            "hierarchy_levels": settings.hierarchy_levels,
            "hierarchy_depth_policy": settings.hierarchy_depth_policy,
            "hierarchy_target_group_size": settings.hierarchy_target_group_size,
            "effective_top_k": hierarchy["top_level_cluster_count"],
            "top_level_cluster_count": hierarchy["top_level_cluster_count"],
            "leaf_cluster_count": hierarchy["leaf_cluster_count"],
            "hierarchy_max_level": hierarchy.get("max_level", 0),
            "hierarchy_generated_levels": int(
                hierarchy.get("generated_levels", hierarchy.get("max_level", 0) + 1)
            ),
            "hierarchy_depth_stop_reason": hierarchy.get("depth_stop_reason"),
            "hierarchy_why_not_deeper": hierarchy.get("why_not_deeper"),
            "hierarchy_label_fallback_count": int(
                hierarchy.get("hierarchy_label_fallback_count", 0)
            ),
            "phase4_hierarchy_execution_mode": phase4_hierarchy_execution_mode,
            "phase4_hierarchy_adaptive_concurrency_enabled": adaptive_concurrency_enabled,
            "hierarchy_label_max_concurrency": hierarchy_label_max_concurrency,
            "hierarchy_label_resume_processed_count": int(
                hierarchy.get("hierarchy_label_resume_count", 0)
            ),
            "hierarchy_label_final_concurrency": int(
                hierarchy.get(
                    "hierarchy_label_final_concurrency",
                    hierarchy_label_max_concurrency if should_use_parallel else 1,
                )
            ),
            "phase4_hierarchy_openai_metrics": phase4_hierarchy_llm_metrics,
            "output_files": output_files,
        }
    )
    _save_run_manifest(run_root, manifest)

    return hierarchy, run_root


def run_phase5_privacy_audit(
    *,
    settings: Settings,
    conversations: list[Conversation],
    facets: list[Facets],
    labeled_clusters: list[dict],
    run_root: Path,
    llm_client: LLMJsonClient | None = None,
    raw_sample_size: int | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[dict, list[dict], Path]:
    """Audit privacy across stages and apply cluster-level privacy gating."""

    if not settings.resolved_openai_api_key() and llm_client is None:
        raise ValueError(
            "OPENAI_API_KEY or AZURE_OPENAI_API_KEY is required for privacy auditing. "
            "Set it in your environment or .env."
        )
    if not facets:
        raise ValueError("No facets provided for privacy auditing.")
    if not labeled_clusters:
        raise ValueError("No labeled clusters provided for privacy auditing.")

    client = llm_client or OpenAIJsonClient(
        api_key=settings.resolved_openai_api_key(),
        model=settings.resolved_openai_model(),
        base_url=settings.resolved_openai_base_url(),
        temperature=settings.openai_temperature,
        max_retries=settings.client_max_retries,
        backoff_seconds=settings.client_backoff_seconds,
    )
    threshold = settings.privacy_threshold_min_rating
    sample_limit = raw_sample_size or settings.privacy_audit_raw_sample_size
    effective_privacy_batch_size = max(1, settings.privacy_batch_size)
    effective_privacy_concurrency = max(1, settings.privacy_max_concurrency)
    should_use_async_batch = llm_client is None and effective_privacy_batch_size > 1
    phase5_execution_mode = "async_batch" if should_use_async_batch else "sync_single"
    adaptive_concurrency_enabled = bool(should_use_async_batch)

    facet_ids = {facet.conversation_id for facet in facets}
    processed_conversations = [
        conversation for conversation in conversations if conversation.conversation_id in facet_ids
    ]
    raw_population = processed_conversations if processed_conversations else conversations
    effective_raw_limit = min(sample_limit, len(raw_population))
    sampled_raw = raw_population[:effective_raw_limit]

    raw_order = [conversation.conversation_id for conversation in sampled_raw]
    facet_order = [facet.conversation_id for facet in facets]
    cluster_order = [str(int(cluster["cluster_id"])) for cluster in labeled_clusters]

    privacy_dir = ensure_directory(run_root / "privacy")
    raw_partial_path = privacy_dir / "raw_conversation.partial.jsonl"
    facet_partial_path = privacy_dir / "facet_summary.partial.jsonl"
    cluster_partial_path = privacy_dir / "cluster_summary.partial.jsonl"
    batch_errors_partial_path = privacy_dir / "batch_errors.partial.jsonl"
    checkpoint_path = privacy_dir / "privacy_checkpoint.json"

    raw_records_by_id: dict[str, dict] = {}
    for row in _load_jsonl_records(raw_partial_path):
        conversation_id = str(row.get("conversation_id", "")).strip()
        if conversation_id in raw_order:
            raw_records_by_id[conversation_id] = row

    facet_records_by_id: dict[str, dict] = {}
    for row in _load_jsonl_records(facet_partial_path):
        conversation_id = str(row.get("conversation_id", "")).strip()
        if conversation_id in facet_order:
            facet_records_by_id[conversation_id] = row

    cluster_records_by_id: dict[str, dict] = {}
    for row in _load_jsonl_records(cluster_partial_path):
        cluster_id = str(row.get("cluster_id", "")).strip()
        if cluster_id in cluster_order:
            cluster_records_by_id[cluster_id] = row

    batch_errors: list[dict] = [
        row for row in _load_jsonl_records(batch_errors_partial_path) if isinstance(row, dict)
    ]

    validation_count = 6 if settings.privacy_validation_enabled else 0
    total_progress = effective_raw_limit + len(facets) + len(labeled_clusters) + validation_count
    progress_offset = 0

    def _emit(done: int, total: int, label: str, offset: int) -> None:
        if progress_callback is None:
            return
        progress_callback(offset + done, max(total_progress, 1), label)

    def _save_privacy_checkpoint(
        *,
        completed: bool,
        note: str,
        current_concurrency: int,
    ) -> None:
        save_json(
            checkpoint_path,
            {
                "phase": "phase5_privacy_audit",
                "run_id": run_root.name,
                "execution_mode": phase5_execution_mode,
                "adaptive_concurrency_enabled": adaptive_concurrency_enabled,
                "privacy_batch_size": effective_privacy_batch_size,
                "privacy_max_concurrency": effective_privacy_concurrency,
                "current_concurrency": current_concurrency,
                "raw_total": effective_raw_limit,
                "raw_processed": len(raw_records_by_id),
                "facet_total": len(facets),
                "facet_processed": len(facet_records_by_id),
                "cluster_total": len(labeled_clusters),
                "cluster_processed": len(cluster_records_by_id),
                "batch_error_count": len(batch_errors),
                "completed": completed,
                "note": note,
                "updated_at_utc": datetime.now(UTC).isoformat(),
            },
        )

    resume_processed_count = (
        len(raw_records_by_id) + len(facet_records_by_id) + len(cluster_records_by_id)
    )
    if progress_callback is not None and resume_processed_count > 0:
        progress_callback(resume_processed_count, total_progress, "audit_resume")

    if should_use_async_batch:
        raw_items: list[tuple[str, str]] = [
            (
                conversation.conversation_id,
                "\n".join(
                    f"{message.role.upper()}: {message.content}"
                    for message in conversation.messages
                ),
            )
            for conversation in sampled_raw
        ]
        facet_items: list[tuple[str, str]] = [
            (
                facet.conversation_id,
                (
                    f"summary={facet.summary}\n"
                    f"task={facet.task}\n"
                    f"language={facet.language}\n"
                    f"turn_count={facet.turn_count}"
                ),
            )
            for facet in facets
        ]
        cluster_items: list[tuple[str, str]] = [
            (str(int(cluster["cluster_id"])), f"{cluster['name']}: {cluster['description']}")
            for cluster in labeled_clusters
        ]

        async def _run_async_privacy_audits() -> tuple[
            dict[str, dict], dict[str, dict], dict[str, dict]
        ]:
            """Run batched privacy audits with adaptive concurrency and checkpoints."""

            async def _audit_stage_items(
                *,
                stage: str,
                items: list[tuple[str, str]],
                progress_label: str,
                offset: int,
                id_field: str,
                partial_path: Path,
                existing_records: dict[str, dict],
                id_cast: Callable[[str], str | int],
            ) -> dict[str, dict]:
                if not items:
                    return {}

                records_by_id = dict(existing_records)
                if records_by_id:
                    _emit(len(records_by_id), len(items), progress_label, offset)

                remaining_items = [
                    (item_id, content) for item_id, content in items if item_id not in records_by_id
                ]
                if not remaining_items:
                    return records_by_id

                batches = [
                    remaining_items[index : index + effective_privacy_batch_size]
                    for index in range(0, len(remaining_items), effective_privacy_batch_size)
                ]
                current_concurrency = min(
                    effective_privacy_concurrency,
                    max(1, len(batches)),
                )
                batch_cursor = 0

                async def _process_batch(
                    batch_items: list[tuple[str, str]],
                ) -> tuple[dict[str, dict], list[dict], bool]:
                    output: dict[str, dict] = {}
                    errors: list[dict] = []
                    had_issue = False
                    expected_ids = {item_id for item_id, _ in batch_items}
                    missing_reason_by_id: dict[str, str] = {}
                    try:
                        batch_results, batch_errors = await asyncio.to_thread(
                            audit_content_batch,
                            stage=stage,
                            items=batch_items,
                            llm_client=client,
                        )
                        output.update(batch_results)
                        for item in batch_errors:
                            error_row = {
                                "stage": stage,
                                "item_id": item.get("content_id"),
                                "error_type": item.get("error_type"),
                                "error": item.get("error"),
                            }
                            errors.append(error_row)
                            content_id = str(item.get("content_id", "")).strip()
                            if content_id in expected_ids:
                                missing_reason_by_id[content_id] = str(
                                    item.get("error", "Batch output missing content_id.")
                                )
                        had_issue = bool(batch_errors)
                    except Exception as exc:
                        had_issue = True
                        errors.append(
                            {
                                "stage": stage,
                                "item_id": None,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                        )
                        for item_id, _ in batch_items:
                            missing_reason_by_id[item_id] = f"Batch privacy audit failed: {exc}"

                    for item_id, content in batch_items:
                        if item_id in output:
                            continue
                        single_result = await asyncio.to_thread(
                            audit_content,
                            stage=stage,
                            content=content,
                            llm_client=client,
                        )
                        missing_reason = missing_reason_by_id.get(item_id)
                        if missing_reason:
                            if single_result.get("error") is None:
                                single_result["error"] = missing_reason
                            single_result["fallback_used"] = True
                            had_issue = True
                        output[item_id] = single_result

                    return output, errors, had_issue

                while batch_cursor < len(batches):
                    wave_batches = batches[batch_cursor : batch_cursor + current_concurrency]
                    batch_cursor += len(wave_batches)

                    tasks = [asyncio.create_task(_process_batch(batch)) for batch in wave_batches]
                    wave_had_issue = False
                    new_rows: list[dict] = []
                    new_batch_errors: list[dict] = []

                    for task in tasks:
                        batch_output, stage_batch_errors, had_issue = await task
                        wave_had_issue = wave_had_issue or had_issue
                        new_batch_errors.extend(stage_batch_errors)

                        for item_id, result in batch_output.items():
                            if item_id in records_by_id:
                                continue
                            row = {
                                "stage": stage,
                                id_field: id_cast(item_id),
                                "rating": int(result["rating"]),
                                "justification": str(result["justification"]),
                                "audit_fallback_used": bool(result.get("fallback_used", False)),
                                "audit_error": result.get("error"),
                            }
                            records_by_id[item_id] = row
                            new_rows.append(row)

                    if new_rows:
                        append_jsonl(partial_path, new_rows)
                    if new_batch_errors:
                        batch_errors.extend(new_batch_errors)
                        append_jsonl(batch_errors_partial_path, new_batch_errors)

                    _emit(len(records_by_id), len(items), progress_label, offset)

                    if wave_had_issue and current_concurrency > 1:
                        current_concurrency -= 1
                    elif (
                        not wave_had_issue
                        and current_concurrency < effective_privacy_concurrency
                        and batch_cursor < len(batches)
                    ):
                        current_concurrency += 1

                    _save_privacy_checkpoint(
                        completed=False,
                        note=f"{stage}_wave",
                        current_concurrency=current_concurrency,
                    )

                return records_by_id

            raw_stage_records = await _audit_stage_items(
                stage="raw_conversation",
                items=raw_items,
                progress_label="audit_raw_conversations",
                offset=0,
                id_field="conversation_id",
                partial_path=raw_partial_path,
                existing_records=raw_records_by_id,
                id_cast=lambda item_id: item_id,
            )
            facet_stage_records = await _audit_stage_items(
                stage="facet_summary",
                items=facet_items,
                progress_label="audit_facets",
                offset=effective_raw_limit,
                id_field="conversation_id",
                partial_path=facet_partial_path,
                existing_records=facet_records_by_id,
                id_cast=lambda item_id: item_id,
            )
            cluster_stage_records = await _audit_stage_items(
                stage="cluster_summary",
                items=cluster_items,
                progress_label="audit_cluster_summaries",
                offset=effective_raw_limit + len(facets),
                id_field="cluster_id",
                partial_path=cluster_partial_path,
                existing_records=cluster_records_by_id,
                id_cast=lambda item_id: int(item_id),
            )
            return raw_stage_records, facet_stage_records, cluster_stage_records

        raw_records_by_id, facet_records_by_id, cluster_records_by_id = asyncio.run(
            _run_async_privacy_audits()
        )
        progress_offset = effective_raw_limit + len(facets) + len(labeled_clusters)
    else:
        if len(raw_records_by_id) < len(raw_order):
            raw_records = audit_raw_conversations(
                conversations=raw_population,
                llm_client=client,
                sample_limit=effective_raw_limit,
                progress_callback=(
                    (
                        lambda done, total, offset=progress_offset: _emit(
                            done, total, "audit_raw_conversations", offset
                        )
                    )
                    if progress_callback is not None
                    else None
                ),
            )
            raw_records_by_id = {str(item["conversation_id"]): item for item in raw_records}
            save_jsonl(raw_partial_path, raw_records)
            _save_privacy_checkpoint(
                completed=False,
                note="raw_completed",
                current_concurrency=1,
            )
        else:
            _emit(
                len(raw_records_by_id),
                len(raw_order),
                "audit_raw_conversations",
                progress_offset,
            )
        progress_offset += effective_raw_limit
        if len(facet_records_by_id) < len(facet_order):
            facet_records = audit_facets(
                facets=facets,
                llm_client=client,
                progress_callback=(
                    (
                        lambda done, total, offset=progress_offset: _emit(
                            done,
                            total,
                            "audit_facets",
                            offset,
                        )
                    )
                    if progress_callback is not None
                    else None
                ),
            )
            facet_records_by_id = {str(item["conversation_id"]): item for item in facet_records}
            save_jsonl(facet_partial_path, facet_records)
            _save_privacy_checkpoint(
                completed=False,
                note="facet_completed",
                current_concurrency=1,
            )
        else:
            _emit(len(facet_records_by_id), len(facet_order), "audit_facets", progress_offset)
        progress_offset += len(facets)
        if len(cluster_records_by_id) < len(cluster_order):
            cluster_records = audit_labeled_clusters(
                labeled_clusters=labeled_clusters,
                llm_client=client,
                progress_callback=(
                    (
                        lambda done, total, offset=progress_offset: _emit(
                            done, total, "audit_cluster_summaries", offset
                        )
                    )
                    if progress_callback is not None
                    else None
                ),
            )
            cluster_records_by_id = {str(int(item["cluster_id"])): item for item in cluster_records}
            save_jsonl(cluster_partial_path, cluster_records)
            _save_privacy_checkpoint(
                completed=False,
                note="cluster_completed",
                current_concurrency=1,
            )
        else:
            _emit(
                len(cluster_records_by_id),
                len(cluster_order),
                "audit_cluster_summaries",
                progress_offset,
            )
        progress_offset += len(labeled_clusters)
    raw_records = [
        raw_records_by_id[item_id] for item_id in raw_order if item_id in raw_records_by_id
    ]
    facet_records = [
        facet_records_by_id[item_id] for item_id in facet_order if item_id in facet_records_by_id
    ]
    cluster_records = [
        cluster_records_by_id[item_id]
        for item_id in cluster_order
        if item_id in cluster_records_by_id
    ]
    gated_clusters = apply_cluster_privacy_gate(
        labeled_clusters=labeled_clusters,
        cluster_audits=cluster_records,
        threshold=threshold,
    )

    summary = {
        "raw_conversation": summarize_privacy_records(raw_records, threshold=threshold),
        "facet_summary": summarize_privacy_records(facet_records, threshold=threshold),
        "cluster_summary": summarize_privacy_records(cluster_records, threshold=threshold),
    }
    validation = (
        run_privacy_auditor_validation(
            client,
            progress_callback=(
                (
                    lambda done, total, offset=progress_offset: _emit(
                        done, total, "audit_validation_set", offset
                    )
                )
                if progress_callback is not None
                else None
            ),
        )
        if settings.privacy_validation_enabled
        else None
    )
    stage_fallback_counts = {
        "raw_conversation": sum(
            1 for item in raw_records if bool(item.get("audit_fallback_used", False))
        ),
        "facet_summary": sum(
            1 for item in facet_records if bool(item.get("audit_fallback_used", False))
        ),
        "cluster_summary": sum(
            1 for item in cluster_records if bool(item.get("audit_fallback_used", False))
        ),
    }
    validation_fallback_count = (
        sum(
            1
            for item in validation.get("records", [])
            if bool(item.get("audit_fallback_used", False))
        )
        if isinstance(validation, dict)
        else 0
    )
    phase5_llm_metrics = _llm_metrics_snapshot(client)

    privacy_dir = ensure_directory(run_root / "privacy")
    clusters_dir = ensure_directory(run_root / "clusters")
    save_json(
        privacy_dir / "privacy_audit.json",
        {
            "threshold": threshold,
            "execution_mode": phase5_execution_mode,
            "adaptive_concurrency_enabled": adaptive_concurrency_enabled,
            "privacy_batch_size": effective_privacy_batch_size,
            "privacy_max_concurrency": effective_privacy_concurrency,
            "privacy_checkpoint_path": str(checkpoint_path.as_posix()),
            "raw_conversation": raw_records,
            "facet_summary": facet_records,
            "cluster_summary": cluster_records,
            "summary": summary,
            "validation": validation,
            "batch_errors": batch_errors,
            "openai_metrics": phase5_llm_metrics,
            "fallback_counts": {
                **stage_fallback_counts,
                "validation": validation_fallback_count,
            },
        },
    )
    save_json(
        clusters_dir / "labeled_clusters_privacy_filtered.json",
        {
            "threshold": threshold,
            "cluster_count_total": len(gated_clusters),
            "cluster_count_kept_by_privacy": sum(
                1 for item in gated_clusters if item["kept_by_privacy"]
            ),
            "cluster_count_final_kept": sum(1 for item in gated_clusters if item["final_kept"]),
            "clusters": gated_clusters,
        },
    )

    _save_privacy_checkpoint(
        completed=True,
        note="completed",
        current_concurrency=(effective_privacy_concurrency if should_use_async_batch else 1),
    )

    assignment_records = _load_jsonl_records(clusters_dir / "base_assignments.jsonl")
    updated_path: Path | None = None
    if assignment_records:
        updated_path = _save_updated_conversations(
            run_root=run_root,
            conversations=conversations,
            facets=facets,
            assignments=assignment_records,
            cluster_labels=gated_clusters,
        )

    manifest = _load_run_manifest(run_root)
    completed_phases = set(manifest.get("completed_phases", []))
    completed_phases.add("phase5_privacy_audit")
    output_files = dict(manifest.get("output_files", {}))
    output_files["privacy_audit_json"] = str((privacy_dir / "privacy_audit.json").as_posix())
    output_files["labeled_clusters_privacy_filtered_json"] = str(
        (clusters_dir / "labeled_clusters_privacy_filtered.json").as_posix()
    )
    output_files["privacy_checkpoint_json"] = str(checkpoint_path.as_posix())
    output_files["privacy_raw_partial_jsonl"] = str(raw_partial_path.as_posix())
    output_files["privacy_facet_partial_jsonl"] = str(facet_partial_path.as_posix())
    output_files["privacy_cluster_partial_jsonl"] = str(cluster_partial_path.as_posix())
    output_files["privacy_batch_errors_partial_jsonl"] = str(batch_errors_partial_path.as_posix())
    if updated_path is not None:
        output_files["conversation_updated_jsonl"] = str(updated_path.as_posix())

    manifest.update(
        {
            "run_id": manifest.get("run_id", run_root.name),
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "phase": "phase5_privacy_audit",
            "completed_phases": sorted(completed_phases),
            "privacy_threshold_min_rating": threshold,
            "privacy_audit_raw_sample_size": effective_raw_limit,
            "privacy_validation_enabled": settings.privacy_validation_enabled,
            "privacy_validation": validation,
            "phase5_execution_mode": phase5_execution_mode,
            "phase5_adaptive_concurrency_enabled": adaptive_concurrency_enabled,
            "privacy_batch_size": effective_privacy_batch_size,
            "privacy_max_concurrency": effective_privacy_concurrency,
            "privacy_batch_error_count": len(batch_errors),
            "privacy_resume_processed_count": resume_processed_count,
            "phase5_openai_metrics": phase5_llm_metrics,
            "privacy_audit_fallback_counts": {
                **stage_fallback_counts,
                "validation": validation_fallback_count,
            },
            "privacy_summary": summary,
            "output_files": output_files,
        }
    )
    _save_run_manifest(run_root, manifest)

    return summary, gated_clusters, run_root


def run_phase6_evaluation(
    *,
    settings: Settings,
    run_root: Path,
    count: int | None = None,
    topic_count: int | None = None,
    language_count: int | None = None,
    seed: int | None = None,
    embedding_client: TextEmbeddingClient | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[dict, Path]:
    """Run synthetic reconstruction evaluation and ablation suite."""

    eval_count = count or settings.eval_synthetic_count
    eval_topic_count = topic_count or settings.eval_topic_count
    eval_language_count = language_count or settings.eval_language_count
    eval_seed = seed or settings.eval_seed

    results = run_synthetic_evaluation_suite(
        settings=settings,
        count=eval_count,
        topic_count=eval_topic_count,
        language_count=eval_language_count,
        seed=eval_seed,
        embedding_client=embedding_client,
        progress_callback=progress_callback,
    )
    eval_fallback_count = sum(
        1
        for item in results.get("ablations", {}).values()
        if isinstance(item, dict) and bool(item.get("fallback_used", False))
    )
    report_md = build_evaluation_markdown_report(results)

    eval_dir = ensure_directory(run_root / "eval")
    save_json(eval_dir / "phase6_metrics.json", results)
    save_jsonl(eval_dir / "synthetic_conversations.jsonl", results["synthetic_conversations"])
    (eval_dir / "report.md").write_text(report_md, encoding="utf-8")

    manifest = _load_run_manifest(run_root)
    completed_phases = set(manifest.get("completed_phases", []))
    completed_phases.add("phase6_evaluation")
    output_files = dict(manifest.get("output_files", {}))
    output_files["phase6_metrics_json"] = str((eval_dir / "phase6_metrics.json").as_posix())
    output_files["phase6_synthetic_jsonl"] = str(
        (eval_dir / "synthetic_conversations.jsonl").as_posix()
    )
    output_files["phase6_report_md"] = str((eval_dir / "report.md").as_posix())

    manifest.update(
        {
            "run_id": manifest.get("run_id", run_root.name),
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "phase": "phase6_evaluation",
            "completed_phases": sorted(completed_phases),
            "phase6": {
                "synthetic_count": eval_count,
                "topic_count": eval_topic_count,
                "language_count": eval_language_count,
                "seed": eval_seed,
                "privacy_summary_accuracy": results["ablations"]["privacy_summary"]["accuracy"],
                "evaluation_fallback_count": eval_fallback_count,
            },
            "output_files": output_files,
        }
    )
    _save_run_manifest(run_root, manifest)

    return results, run_root

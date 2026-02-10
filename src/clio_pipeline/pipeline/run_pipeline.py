"""Orchestration helpers for pipeline phases."""

from __future__ import annotations

import json
import secrets
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from clio_pipeline.config import Settings
from clio_pipeline.io import (
    DatasetSummary,
    ensure_directory,
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
from clio_pipeline.pipeline.clustering import build_base_cluster_outputs, fit_base_kmeans
from clio_pipeline.pipeline.embedding import embed_texts_in_batches
from clio_pipeline.pipeline.evaluate import (
    build_evaluation_markdown_report,
    run_synthetic_evaluation_suite,
)
from clio_pipeline.pipeline.facet_extraction import extract_conversation_facets
from clio_pipeline.pipeline.hierarchy import build_multilevel_hierarchy_scaffold
from clio_pipeline.pipeline.labeling import label_clusters
from clio_pipeline.pipeline.privacy_audit import (
    apply_cluster_privacy_gate,
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
) -> tuple[str, Path]:
    """Create run folder and seed base artifacts."""

    effective_run_id = run_id or generate_run_id()
    run_root = ensure_directory(settings.output_dir / effective_run_id)
    conversation_path = save_jsonl(
        run_root / "conversation.jsonl",
        [_serialize_messages_only(conversation) for conversation in conversations],
    )
    updated_path = _save_updated_conversations(
        run_root=run_root,
        conversations=conversations,
    )

    manifest = _load_run_manifest(run_root)
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
            "output_files": output_files,
        }
    )
    _save_run_manifest(run_root, manifest)
    return effective_run_id, run_root


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

    client = llm_client or OpenAIJsonClient(
        api_key=settings.resolved_openai_api_key(),
        model=settings.resolved_openai_model(),
        base_url=settings.resolved_openai_base_url(),
        temperature=settings.openai_temperature,
        max_retries=settings.client_max_retries,
        backoff_seconds=settings.client_backoff_seconds,
    )
    facets: list[Facets] = []
    extraction_errors: list[dict] = []
    total = len(selected_conversations)
    for index, conversation in enumerate(selected_conversations, start=1):
        try:
            facets.append(extract_conversation_facets(conversation, client))
        except Exception as exc:
            extraction_errors.append(
                {
                    "conversation_id": conversation.conversation_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        if progress_callback is not None:
            progress_callback(index, total, "extract_facets")

    if not facets:
        raise ValueError("No facets were extracted successfully; aborting Phase 2.")

    save_jsonl(facets_dir / "facets.jsonl", facets)
    if extraction_errors:
        save_jsonl(facets_dir / "facets_errors.jsonl", extraction_errors)
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
            "output_files": output_files,
        }
    )
    _save_run_manifest(run_root, manifest)

    return facets, run_root


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

    labels, centroids, effective_k = fit_base_kmeans(
        embeddings,
        requested_k=settings.k_base_clusters,
        random_seed=settings.random_seed,
    )
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
            "effective_k": effective_k,
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
            "requested_k": settings.k_base_clusters,
            "effective_k": effective_k,
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
    labeled_clusters = label_clusters(
        cluster_summaries=cluster_summaries,
        facets=facets,
        llm_client=client,
        sample_size=effective_sample_size,
        progress_callback=(
            (lambda done, total: progress_callback(done, total, "label_clusters"))
            if progress_callback is not None
            else None
        ),
    )
    labeling_errors = [
        {
            "cluster_id": int(item["cluster_id"]),
            "error": str(item.get("labeling_error", "")),
        }
        for item in labeled_clusters
        if bool(item.get("labeling_fallback_used", False))
    ]

    clusters_dir = ensure_directory(run_root / "clusters")
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
    effective_batch_size = batch_size or settings.embedding_batch_size

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

    hierarchy = build_multilevel_hierarchy_scaffold(
        labeled_clusters=labeled_clusters,
        leaf_embeddings=embeddings,
        llm_client=llm,
        max_levels=settings.hierarchy_levels,
        target_group_size=settings.hierarchy_target_group_size,
        random_seed=settings.random_seed,
        progress_callback=(
            (lambda done, total: progress_callback(done, total, "label_hierarchy_groups"))
            if progress_callback is not None
            else None
        ),
    )
    hierarchy["requested_top_k"] = settings.hierarchy_top_k

    clusters_dir = ensure_directory(run_root / "clusters")
    viz_dir = ensure_directory(run_root / "viz")
    save_json(clusters_dir / "hierarchy.json", hierarchy)
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
    output_files["tree_view_json"] = str((viz_dir / "tree_view.json").as_posix())
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
            "hierarchy_target_group_size": settings.hierarchy_target_group_size,
            "effective_top_k": hierarchy["top_level_cluster_count"],
            "top_level_cluster_count": hierarchy["top_level_cluster_count"],
            "leaf_cluster_count": hierarchy["leaf_cluster_count"],
            "hierarchy_max_level": hierarchy.get("max_level", 0),
            "hierarchy_label_fallback_count": int(
                hierarchy.get("hierarchy_label_fallback_count", 0)
            ),
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

    facet_ids = {facet.conversation_id for facet in facets}
    processed_conversations = [
        conversation for conversation in conversations if conversation.conversation_id in facet_ids
    ]
    raw_population = processed_conversations if processed_conversations else conversations
    effective_raw_limit = min(sample_limit, len(raw_population))
    validation_count = 6 if settings.privacy_validation_enabled else 0
    total_progress = effective_raw_limit + len(facets) + len(labeled_clusters) + validation_count
    progress_offset = 0

    def _emit(done: int, total: int, label: str, offset: int) -> None:
        if progress_callback is None:
            return
        progress_callback(offset + done, max(total_progress, 1), label)

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
    progress_offset += effective_raw_limit
    facet_records = audit_facets(
        facets=facets,
        llm_client=client,
        progress_callback=(
            (lambda done, total, offset=progress_offset: _emit(done, total, "audit_facets", offset))
            if progress_callback is not None
            else None
        ),
    )
    progress_offset += len(facets)
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
    progress_offset += len(labeled_clusters)
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

    privacy_dir = ensure_directory(run_root / "privacy")
    clusters_dir = ensure_directory(run_root / "clusters")
    save_json(
        privacy_dir / "privacy_audit.json",
        {
            "threshold": threshold,
            "raw_conversation": raw_records,
            "facet_summary": facet_records,
            "cluster_summary": cluster_records,
            "summary": summary,
            "validation": validation,
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

"""Cluster labeling logic for Phase 4."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, ConfigDict, Field

from clio_pipeline.models import LLMJsonClient
from clio_pipeline.prompts import CLUSTER_LABEL_SYSTEM_PROMPT, build_cluster_label_user_prompt
from clio_pipeline.schemas import Facets


class ClusterLabelingError(ValueError):
    """Raised when cluster labeling payloads fail validation."""


class _ClusterLabelPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=1, max_length=500)


def _build_fallback_label(cluster: dict, exc: Exception) -> dict:
    """Construct a safe fallback label when LLM labeling fails."""

    cluster_id = int(cluster["cluster_id"])
    size = int(cluster["size"])
    unique_users = int(cluster["unique_users"])
    return {
        "name": f"Cluster {cluster_id}",
        "description": (
            "Fallback label due to labeling error. "
            f"Cluster has {size} conversations and {unique_users} unique users."
        ),
        "labeling_fallback_used": True,
        "labeling_error": str(exc),
    }


def _label_one_cluster(
    *,
    cluster: dict,
    facet_by_id: dict[str, Facets],
    llm_client: LLMJsonClient,
    sample_size: int,
) -> dict:
    """Generate one labeled cluster row with fallback handling."""

    conversation_ids = [str(item) for item in cluster.get("conversation_ids", [])]
    summaries: list[str] = []
    for conversation_id in conversation_ids:
        facet = facet_by_id.get(conversation_id)
        if facet is not None:
            summaries.append(facet.summary)
        if len(summaries) >= sample_size:
            break

    if not summaries:
        summaries = ["No sample summary available for this cluster."]

    try:
        payload = llm_client.complete_json(
            system_prompt=CLUSTER_LABEL_SYSTEM_PROMPT,
            user_prompt=build_cluster_label_user_prompt(
                cluster_id=int(cluster["cluster_id"]),
                size=int(cluster["size"]),
                unique_users=int(cluster["unique_users"]),
                summaries=summaries,
            ),
            schema_name="cluster_label_payload",
            json_schema=_ClusterLabelPayload.model_json_schema(),
            strict_schema=True,
        )
        parsed = _ClusterLabelPayload.model_validate(payload)
        name = parsed.name.strip()
        description = parsed.description.strip()
        labeling_fallback_used = False
        labeling_error = None
    except Exception as exc:
        fallback = _build_fallback_label(cluster, exc)
        name = fallback["name"]
        description = fallback["description"]
        labeling_fallback_used = bool(fallback["labeling_fallback_used"])
        labeling_error = str(fallback["labeling_error"])

    return {
        "cluster_id": int(cluster["cluster_id"]),
        "name": name,
        "description": description,
        "size": int(cluster["size"]),
        "unique_users": int(cluster["unique_users"]),
        "kept_by_threshold": bool(cluster["kept_by_threshold"]),
        "conversation_ids": conversation_ids,
        "labeling_fallback_used": labeling_fallback_used,
        "labeling_error": labeling_error,
    }


def label_clusters(
    *,
    cluster_summaries: list[dict],
    facets: list[Facets],
    llm_client: LLMJsonClient,
    sample_size: int = 12,
    max_concurrency: int = 1,
    progress_callback: Callable[[int, int], None] | None = None,
    executor: ThreadPoolExecutor | None = None,
) -> list[dict]:
    """Generate name/description labels for each base cluster summary."""

    if sample_size <= 0:
        raise ClusterLabelingError(f"sample_size must be positive, got {sample_size}.")
    if max_concurrency <= 0:
        raise ClusterLabelingError(f"max_concurrency must be positive, got {max_concurrency}.")

    facet_by_id = {facet.conversation_id: facet for facet in facets}
    ordered_clusters = sorted(cluster_summaries, key=lambda item: int(item["cluster_id"]))
    if max_concurrency == 1 or len(ordered_clusters) <= 1:
        labeled: list[dict] = []
        for index, cluster in enumerate(ordered_clusters, start=1):
            labeled.append(
                _label_one_cluster(
                    cluster=cluster,
                    facet_by_id=facet_by_id,
                    llm_client=llm_client,
                    sample_size=sample_size,
                )
            )
            if progress_callback is not None:
                progress_callback(index, len(ordered_clusters))
        return labeled

    labeled_by_id: dict[int, dict] = {}
    total = len(ordered_clusters)
    done = 0

    def _run_with_executor(pool: ThreadPoolExecutor) -> None:
        nonlocal done
        future_to_cluster_id = {
            pool.submit(
                _label_one_cluster,
                cluster=cluster,
                facet_by_id=facet_by_id,
                llm_client=llm_client,
                sample_size=sample_size,
            ): int(cluster["cluster_id"])
            for cluster in ordered_clusters
        }
        for future in as_completed(future_to_cluster_id):
            labeled = future.result()
            labeled_by_id[int(labeled["cluster_id"])] = labeled
            done += 1
            if progress_callback is not None:
                progress_callback(done, total)

    if executor is not None:
        _run_with_executor(executor)
    else:
        with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
            _run_with_executor(pool)

    return [labeled_by_id[int(cluster["cluster_id"])] for cluster in ordered_clusters]

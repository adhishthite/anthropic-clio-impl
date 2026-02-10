"""Hierarchy scaffolding for Phase 4."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sklearn.cluster import KMeans

from clio_pipeline.models import LLMJsonClient
from clio_pipeline.prompts import HIERARCHY_LABEL_SYSTEM_PROMPT, build_hierarchy_label_user_prompt


class HierarchyError(ValueError):
    """Raised when hierarchy construction inputs are invalid."""


class _HierarchyLabelPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=1, max_length=500)


def fit_hierarchy_groups(
    embeddings: np.ndarray,
    *,
    requested_top_k: int,
    random_seed: int,
) -> tuple[np.ndarray, int]:
    """Assign each leaf cluster to a top-level group."""

    if embeddings.ndim != 2:
        raise HierarchyError(f"Embeddings must be 2D, got ndim={embeddings.ndim}.")
    if embeddings.shape[0] == 0:
        raise HierarchyError("Hierarchy embeddings cannot be empty.")
    if requested_top_k <= 0:
        raise HierarchyError(f"requested_top_k must be positive, got {requested_top_k}.")

    effective_top_k = min(requested_top_k, embeddings.shape[0])
    if effective_top_k == 1:
        labels = np.zeros(embeddings.shape[0], dtype=int)
        return labels, effective_top_k

    model = KMeans(n_clusters=effective_top_k, n_init=10, random_state=random_seed)
    labels = model.fit_predict(embeddings).astype(int)
    return labels, effective_top_k


def build_hierarchy_scaffold(
    *,
    labeled_clusters: list[dict],
    group_labels: np.ndarray,
    llm_client: LLMJsonClient,
) -> dict:
    """Build a two-level hierarchy from labeled clusters."""

    if len(labeled_clusters) != len(group_labels):
        raise HierarchyError(
            "labeled cluster count does not match group_labels length: "
            f"{len(labeled_clusters)} != {len(group_labels)}."
        )

    grouped_children: dict[int, list[dict]] = {}
    for cluster, group_label in zip(labeled_clusters, group_labels, strict=True):
        grouped_children.setdefault(int(group_label), []).append(cluster)

    top_level_clusters: list[dict] = []
    leaf_clusters: list[dict] = []

    for top_index, group_label in enumerate(sorted(grouped_children.keys())):
        children = grouped_children[group_label]
        payload = llm_client.complete_json(
            system_prompt=HIERARCHY_LABEL_SYSTEM_PROMPT,
            user_prompt=build_hierarchy_label_user_prompt(
                group_index=group_label,
                child_clusters=children,
            ),
            schema_name="hierarchy_label_payload",
            json_schema=_HierarchyLabelPayload.model_json_schema(),
            strict_schema=True,
        )
        try:
            parsed = _HierarchyLabelPayload.model_validate(payload)
        except Exception as exc:
            raise HierarchyError(
                f"Hierarchy label payload failed validation for group {group_label}: {exc}"
            ) from exc

        top_cluster_id = f"top-{top_index:03d}"
        child_ids = [int(item["cluster_id"]) for item in children]
        top_level_clusters.append(
            {
                "top_cluster_id": top_cluster_id,
                "name": parsed.name.strip(),
                "description": parsed.description.strip(),
                "child_cluster_ids": child_ids,
                "child_count": len(child_ids),
            }
        )

        for child in children:
            leaf = dict(child)
            leaf["top_cluster_id"] = top_cluster_id
            leaf_clusters.append(leaf)

    return {
        "top_level_cluster_count": len(top_level_clusters),
        "leaf_cluster_count": len(leaf_clusters),
        "top_level_clusters": top_level_clusters,
        "leaf_clusters": leaf_clusters,
    }


def _validate_label_payload(payload: dict, *, group_index: int) -> _HierarchyLabelPayload:
    """Validate one hierarchy label payload."""

    try:
        return _HierarchyLabelPayload.model_validate(payload)
    except Exception as exc:
        raise HierarchyError(
            f"Hierarchy label payload failed validation for group {group_index}: {exc}"
        ) from exc


def build_multilevel_hierarchy_scaffold(
    *,
    labeled_clusters: list[dict],
    leaf_embeddings: np.ndarray,
    llm_client: LLMJsonClient,
    max_levels: int,
    target_group_size: int,
    random_seed: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    """Build a multi-level hierarchy from labeled clusters."""

    if len(labeled_clusters) == 0:
        raise HierarchyError("No labeled clusters were provided.")
    if len(labeled_clusters) != leaf_embeddings.shape[0]:
        raise HierarchyError(
            "labeled cluster count does not match leaf embeddings rows: "
            f"{len(labeled_clusters)} != {leaf_embeddings.shape[0]}."
        )
    if max_levels <= 0:
        raise HierarchyError(f"max_levels must be positive, got {max_levels}.")
    if target_group_size <= 1:
        raise HierarchyError(
            f"target_group_size must be > 1 to reduce hierarchy width, got {target_group_size}."
        )

    nodes: list[dict] = []
    edges: list[dict] = []
    current_nodes: list[dict] = []
    current_embeddings = leaf_embeddings
    label_ops_done = 0
    total_label_ops = 0
    label_fallback_records: list[dict] = []

    simulated_nodes = len(labeled_clusters)
    for _ in range(max_levels):
        simulated_groups = max(1, int(np.ceil(simulated_nodes / target_group_size)))
        if simulated_groups >= simulated_nodes:
            break
        total_label_ops += simulated_groups
        simulated_nodes = simulated_groups

    for cluster in labeled_clusters:
        leaf_node = {
            "node_id": f"leaf-{int(cluster['cluster_id']):03d}",
            "level": 0,
            "name": str(cluster["name"]).strip(),
            "description": str(cluster["description"]).strip(),
            "size": int(cluster["size"]),
            "source_cluster_id": int(cluster["cluster_id"]),
            "child_ids": [],
        }
        nodes.append(leaf_node)
        current_nodes.append(leaf_node)

    level = 1
    while level <= max_levels and len(current_nodes) > 1:
        requested_groups = max(1, int(np.ceil(len(current_nodes) / target_group_size)))
        group_labels, effective_groups = fit_hierarchy_groups(
            current_embeddings,
            requested_top_k=requested_groups,
            random_seed=random_seed + level,
        )
        if effective_groups >= len(current_nodes):
            break

        grouped_children: dict[int, list[dict]] = {}
        grouped_indexes: dict[int, list[int]] = {}
        for index, (node, group_label) in enumerate(zip(current_nodes, group_labels, strict=True)):
            group_id = int(group_label)
            grouped_children.setdefault(group_id, []).append(node)
            grouped_indexes.setdefault(group_id, []).append(index)

        next_nodes: list[dict] = []
        next_embeddings_rows: list[np.ndarray] = []
        for group_offset, group_id in enumerate(sorted(grouped_children.keys())):
            children = grouped_children[group_id]
            child_clusters = [
                {
                    "cluster_id": int(child["source_cluster_id"])
                    if child["source_cluster_id"] is not None
                    else group_offset,
                    "name": child["name"],
                    "description": child["description"],
                    "size": int(child["size"]),
                    "unique_users": int(child["size"]),
                }
                for child in children
            ]
            try:
                payload = llm_client.complete_json(
                    system_prompt=HIERARCHY_LABEL_SYSTEM_PROMPT,
                    user_prompt=build_hierarchy_label_user_prompt(
                        group_index=group_id,
                        child_clusters=child_clusters,
                    ),
                    schema_name="hierarchy_label_payload",
                    json_schema=_HierarchyLabelPayload.model_json_schema(),
                    strict_schema=True,
                )
                parsed = _validate_label_payload(payload, group_index=group_id)
                group_name = parsed.name.strip()
                group_description = parsed.description.strip()
                fallback_used = False
                fallback_error = None
            except Exception as exc:
                group_name = f"Group L{level}-{group_offset}"
                group_description = (
                    "Fallback hierarchy label due to model error. "
                    f"Aggregates {len(children)} child nodes."
                )
                fallback_used = True
                fallback_error = str(exc)
                label_fallback_records.append(
                    {
                        "level": level,
                        "group_offset": group_offset,
                        "group_id": group_id,
                        "error": str(exc),
                    }
                )
            label_ops_done += 1
            if progress_callback is not None and total_label_ops > 0:
                progress_callback(label_ops_done, total_label_ops)

            node_id = f"lvl-{level:02d}-{group_offset:03d}"
            node = {
                "node_id": node_id,
                "level": level,
                "name": group_name,
                "description": group_description,
                "size": int(sum(int(item["size"]) for item in children)),
                "source_cluster_id": None,
                "child_ids": [child["node_id"] for child in children],
                "hierarchy_label_fallback_used": fallback_used,
                "hierarchy_label_error": fallback_error,
            }
            nodes.append(node)
            next_nodes.append(node)

            for child in children:
                edges.append({"parent_id": node_id, "child_id": child["node_id"]})

            index_rows = grouped_indexes[group_id]
            next_embeddings_rows.append(np.mean(current_embeddings[index_rows], axis=0))

        current_nodes = next_nodes
        current_embeddings = np.stack(next_embeddings_rows, axis=0)
        level += 1

    by_id = {node["node_id"]: node for node in nodes}
    children_by_parent: dict[str, list[str]] = {}
    parent_by_child: dict[str, str] = {}
    for edge in edges:
        children_by_parent.setdefault(edge["parent_id"], []).append(edge["child_id"])
        parent_by_child[edge["child_id"]] = edge["parent_id"]

    top_nodes = [node for node in current_nodes]

    def _collect_leaf_cluster_ids(node_id: str) -> list[int]:
        stack = [node_id]
        leaf_ids: list[int] = []
        while stack:
            current = stack.pop()
            child_ids = children_by_parent.get(current, [])
            if not child_ids:
                cluster_id = by_id[current]["source_cluster_id"]
                if cluster_id is not None:
                    leaf_ids.append(int(cluster_id))
                continue
            stack.extend(child_ids)
        return sorted(leaf_ids)

    top_level_clusters: list[dict] = []
    for node in top_nodes:
        child_cluster_ids = _collect_leaf_cluster_ids(node["node_id"])
        top_level_clusters.append(
            {
                "top_cluster_id": node["node_id"],
                "name": node["name"],
                "description": node["description"],
                "child_cluster_ids": child_cluster_ids,
                "child_count": len(child_cluster_ids),
            }
        )

    leaf_clusters: list[dict] = []
    for cluster in labeled_clusters:
        cluster_id = int(cluster["cluster_id"])
        node_id = f"leaf-{cluster_id:03d}"
        current = node_id
        while current in parent_by_child:
            current = parent_by_child[current]
        leaf = dict(cluster)
        leaf["top_cluster_id"] = current
        leaf_clusters.append(leaf)

    max_level = max((int(node["level"]) for node in nodes), default=0)
    return {
        "top_level_cluster_count": len(top_level_clusters),
        "leaf_cluster_count": len(leaf_clusters),
        "top_level_clusters": top_level_clusters,
        "leaf_clusters": leaf_clusters,
        "max_level": max_level,
        "nodes": nodes,
        "edges": edges,
        "hierarchy_label_fallback_count": len(label_fallback_records),
        "hierarchy_label_fallback_records": label_fallback_records,
    }

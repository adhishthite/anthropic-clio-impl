"""Base clustering utilities for Phase 3."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from clio_pipeline.schemas import Conversation, Facets


class ClusteringError(ValueError):
    """Raised when clustering inputs are invalid."""


def fit_base_kmeans(
    embeddings: np.ndarray,
    *,
    requested_k: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit k-means and return labels, centroids, and effective k."""

    if embeddings.ndim != 2:
        raise ClusteringError(f"Embeddings must be 2D, got ndim={embeddings.ndim}.")
    if embeddings.shape[0] == 0:
        raise ClusteringError("Embeddings cannot be empty.")
    if requested_k <= 0:
        raise ClusteringError(f"requested_k must be positive, got {requested_k}.")

    effective_k = min(requested_k, embeddings.shape[0])
    model = KMeans(n_clusters=effective_k, n_init=10, random_state=random_seed)
    labels = model.fit_predict(embeddings)
    return labels.astype(int), model.cluster_centers_, effective_k


def build_base_cluster_outputs(
    *,
    conversations: list[Conversation],
    facets: list[Facets],
    labels: np.ndarray,
    min_unique_users: int,
    min_conversations_per_cluster: int,
) -> tuple[list[dict], list[dict]]:
    """Build cluster summaries and per-conversation assignment rows."""

    if len(facets) != len(labels):
        raise ClusteringError(
            f"Facet count and label count mismatch: {len(facets)} != {len(labels)}."
        )

    conversations_by_id = {
        conversation.conversation_id: conversation for conversation in conversations
    }
    cluster_users: dict[int, set[str]] = {}
    cluster_conversation_ids: dict[int, list[str]] = {}

    assignments: list[dict] = []
    for facet, cluster_id in zip(facets, labels, strict=True):
        conversation = conversations_by_id.get(facet.conversation_id)
        if conversation is None:
            raise ClusteringError(
                f"Facet references unknown conversation_id '{facet.conversation_id}'."
            )

        cid = int(cluster_id)
        cluster_users.setdefault(cid, set()).add(conversation.user_id)
        cluster_conversation_ids.setdefault(cid, []).append(conversation.conversation_id)
        assignments.append(
            {
                "conversation_id": conversation.conversation_id,
                "user_id": conversation.user_id,
                "cluster_id": cid,
            }
        )

    cluster_summaries: list[dict] = []
    kept_by_cluster: dict[int, bool] = {}
    for cluster_id in sorted(cluster_conversation_ids.keys()):
        size = len(cluster_conversation_ids[cluster_id])
        unique_users = len(cluster_users[cluster_id])
        kept = size >= min_conversations_per_cluster and unique_users >= min_unique_users
        kept_by_cluster[cluster_id] = kept
        cluster_summaries.append(
            {
                "cluster_id": cluster_id,
                "size": size,
                "unique_users": unique_users,
                "kept_by_threshold": kept,
                "conversation_ids": cluster_conversation_ids[cluster_id],
            }
        )

    for assignment in assignments:
        assignment["kept_by_threshold"] = kept_by_cluster[assignment["cluster_id"]]

    return cluster_summaries, assignments

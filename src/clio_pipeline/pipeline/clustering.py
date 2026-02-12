"""Base clustering utilities for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from clio_pipeline.schemas import Conversation, Facets


class ClusteringError(ValueError):
    """Raised when clustering inputs are invalid."""


SUPPORTED_CLUSTER_STRATEGIES = {"kmeans", "hdbscan", "hybrid"}
SUPPORTED_CLUSTER_LEAF_MODES = {"fixed", "auto"}
SUPPORTED_CLUSTER_NOISE_POLICIES = {"nearest", "singleton", "drop"}


@dataclass(frozen=True, slots=True)
class BaseClusterTarget:
    """Resolved cluster target for phase-3."""

    requested_k: int
    leaf_mode: str
    auto_target_k: int | None


@dataclass(frozen=True, slots=True)
class BaseClusteringResult:
    """Structured clustering result for phase-3."""

    labels: np.ndarray
    centroids: np.ndarray
    effective_k: int
    requested_k: int
    strategy: str
    leaf_mode: str
    noise_policy: str
    noise_count: int
    raw_cluster_count: int
    refinement_splits: int
    silhouette_score: float | None
    davies_bouldin_score: float | None
    calinski_harabasz_score: float | None
    fallback_reason: str | None
    auto_target_k: int | None


def resolve_base_cluster_target(
    *,
    sample_count: int,
    requested_k: int,
    leaf_mode: str,
    target_leaf_size: int,
    min_leaf_clusters: int,
    max_leaf_clusters: int,
) -> BaseClusterTarget:
    """Resolve phase-3 target cluster count from fixed or auto policy."""

    if sample_count <= 0:
        raise ClusteringError(f"sample_count must be positive, got {sample_count}.")
    if requested_k <= 0:
        raise ClusteringError(f"requested_k must be positive, got {requested_k}.")
    normalized_leaf_mode = leaf_mode.strip().lower()
    if normalized_leaf_mode not in SUPPORTED_CLUSTER_LEAF_MODES:
        allowed = ", ".join(sorted(SUPPORTED_CLUSTER_LEAF_MODES))
        raise ClusteringError(f"Unsupported leaf_mode '{leaf_mode}'. Expected one of: {allowed}.")

    if normalized_leaf_mode == "fixed":
        return BaseClusterTarget(
            requested_k=max(1, min(sample_count, requested_k)),
            leaf_mode=normalized_leaf_mode,
            auto_target_k=None,
        )

    if target_leaf_size <= 0:
        raise ClusteringError(f"target_leaf_size must be positive, got {target_leaf_size}.")
    if min_leaf_clusters <= 0:
        raise ClusteringError(f"min_leaf_clusters must be positive, got {min_leaf_clusters}.")
    if max_leaf_clusters <= 0:
        raise ClusteringError(f"max_leaf_clusters must be positive, got {max_leaf_clusters}.")
    if min_leaf_clusters > max_leaf_clusters:
        raise ClusteringError(
            "min_leaf_clusters cannot exceed max_leaf_clusters: "
            f"{min_leaf_clusters} > {max_leaf_clusters}."
        )

    auto_target = ceil(sample_count / target_leaf_size)
    bounded_target = max(min_leaf_clusters, min(max_leaf_clusters, auto_target))
    return BaseClusterTarget(
        requested_k=max(1, min(sample_count, bounded_target)),
        leaf_mode=normalized_leaf_mode,
        auto_target_k=bounded_target,
    )


def _validate_embeddings(embeddings: np.ndarray) -> None:
    """Validate embedding matrix shape."""

    if embeddings.ndim != 2:
        raise ClusteringError(f"Embeddings must be 2D, got ndim={embeddings.ndim}.")
    if embeddings.shape[0] == 0:
        raise ClusteringError("Embeddings cannot be empty.")


def fit_base_kmeans(
    embeddings: np.ndarray,
    *,
    requested_k: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit k-means and return labels, centroids, and effective k."""

    _validate_embeddings(embeddings)
    if requested_k <= 0:
        raise ClusteringError(f"requested_k must be positive, got {requested_k}.")

    effective_k = min(requested_k, embeddings.shape[0])
    model = KMeans(n_clusters=effective_k, n_init=10, random_state=random_seed)
    labels = model.fit_predict(embeddings)
    return labels.astype(int), model.cluster_centers_, effective_k


def _normalize_labels(labels: np.ndarray) -> np.ndarray:
    """Map arbitrary integer labels to contiguous cluster IDs [0..k-1]."""

    unique = sorted({int(label) for label in labels.tolist()})
    mapping = {label: index for index, label in enumerate(unique)}
    return np.array([mapping[int(label)] for label in labels.tolist()], dtype=int)


def _centroids_from_labels(embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute centroid matrix from assignments."""

    unique = sorted({int(label) for label in labels.tolist()})
    if not unique:
        raise ClusteringError("At least one cluster label is required.")
    return np.stack(
        [np.mean(embeddings[labels == label], axis=0) for label in unique],
        axis=0,
    )


def _assign_noise_by_nearest_centroid(embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Assign HDBSCAN noise points to nearest non-noise centroid."""

    noise_mask = labels < 0
    if not np.any(noise_mask):
        return labels

    non_noise_mask = ~noise_mask
    non_noise_labels = sorted({int(label) for label in labels[non_noise_mask].tolist()})
    if not non_noise_labels:
        return labels

    centroids = np.stack(
        [np.mean(embeddings[labels == label], axis=0) for label in non_noise_labels],
        axis=0,
    )
    noise_embeddings = embeddings[noise_mask]
    deltas = noise_embeddings[:, None, :] - centroids[None, :, :]
    nearest_indexes = np.argmin(np.sum(deltas * deltas, axis=2), axis=1)
    assigned = labels.copy()
    assigned[noise_mask] = np.array(
        [non_noise_labels[int(index)] for index in nearest_indexes],
        dtype=int,
    )
    return assigned


def _assign_noise_singleton(labels: np.ndarray) -> np.ndarray:
    """Assign each noise point to its own singleton cluster."""

    assigned = labels.copy()
    noise_indexes = np.flatnonzero(assigned < 0)
    if noise_indexes.size == 0:
        return assigned

    existing = [int(label) for label in assigned.tolist() if int(label) >= 0]
    next_label = (max(existing) + 1) if existing else 0
    for noise_index in noise_indexes:
        assigned[int(noise_index)] = next_label
        next_label += 1
    return assigned


def _assign_noise_drop_bucket(labels: np.ndarray) -> np.ndarray:
    """Assign all noise points to one drop bucket cluster."""

    assigned = labels.copy()
    noise_mask = assigned < 0
    if not np.any(noise_mask):
        return assigned
    existing = [int(label) for label in assigned.tolist() if int(label) >= 0]
    drop_bucket_label = (max(existing) + 1) if existing else 0
    assigned[noise_mask] = drop_bucket_label
    return assigned


def _apply_noise_policy(
    *,
    embeddings: np.ndarray,
    labels: np.ndarray,
    noise_policy: str,
) -> tuple[np.ndarray, str]:
    """Apply configured policy to HDBSCAN noise assignments."""

    normalized_noise_policy = noise_policy.strip().lower()
    if normalized_noise_policy not in SUPPORTED_CLUSTER_NOISE_POLICIES:
        allowed = ", ".join(sorted(SUPPORTED_CLUSTER_NOISE_POLICIES))
        raise ClusteringError(
            f"Unsupported clustering_noise_policy '{noise_policy}'. Expected one of: {allowed}."
        )

    if normalized_noise_policy == "nearest":
        resolved = _assign_noise_by_nearest_centroid(embeddings, labels)
        if np.any(resolved < 0):
            resolved = _assign_noise_singleton(resolved)
            return resolved, "nearest_with_singleton_fallback"
        return resolved, normalized_noise_policy
    if normalized_noise_policy == "singleton":
        return _assign_noise_singleton(labels), normalized_noise_policy
    return _assign_noise_drop_bucket(labels), "drop_bucket"


def _refine_cluster_count_hybrid(
    *,
    embeddings: np.ndarray,
    labels: np.ndarray,
    target_k: int,
    random_seed: int,
) -> tuple[np.ndarray, int]:
    """Split largest clusters with local k-means until target_k or no viable split."""

    refined = labels.copy()
    split_count = 0

    while len({int(label) for label in refined.tolist()}) < target_k:
        unique = sorted({int(label) for label in refined.tolist()})
        candidate_label = -1
        candidate_size = 0
        for label in unique:
            size = int(np.count_nonzero(refined == label))
            if size > candidate_size:
                candidate_label = label
                candidate_size = size

        if candidate_label < 0 or candidate_size < 2:
            break

        member_indexes = np.flatnonzero(refined == candidate_label)
        if member_indexes.size < 2:
            break

        sub_embeddings = embeddings[member_indexes]
        sub_model = KMeans(
            n_clusters=2,
            n_init=10,
            random_state=random_seed + split_count + 1,
        )
        sub_labels = sub_model.fit_predict(sub_embeddings).astype(int)
        if len({int(label) for label in sub_labels.tolist()}) < 2:
            break

        next_label = max(unique) + 1
        refined[member_indexes[sub_labels == 1]] = next_label
        split_count += 1

    return refined, split_count


def _safe_clustering_scores(
    *,
    embeddings: np.ndarray,
    labels: np.ndarray,
    random_seed: int,
) -> tuple[float | None, float | None, float | None]:
    """Compute clustering quality metrics when valid."""

    unique_count = len({int(label) for label in labels.tolist()})
    sample_count = embeddings.shape[0]
    if unique_count <= 1 or sample_count <= unique_count:
        return None, None, None

    silhouette_value: float | None = None
    davies_bouldin_value: float | None = None
    calinski_harabasz_value: float | None = None
    try:
        silhouette_value = float(
            silhouette_score(
                embeddings,
                labels,
                sample_size=min(sample_count, 2000),
                random_state=random_seed,
            )
        )
    except Exception:
        silhouette_value = None
    try:
        davies_bouldin_value = float(davies_bouldin_score(embeddings, labels))
    except Exception:
        davies_bouldin_value = None
    try:
        calinski_harabasz_value = float(calinski_harabasz_score(embeddings, labels))
    except Exception:
        calinski_harabasz_value = None
    return silhouette_value, davies_bouldin_value, calinski_harabasz_value


def fit_base_clusters(
    embeddings: np.ndarray,
    *,
    strategy: str,
    leaf_mode: str,
    requested_k: int,
    target_leaf_size: int,
    min_leaf_clusters: int,
    max_leaf_clusters: int,
    hdbscan_min_cluster_size: int,
    hdbscan_min_samples: int,
    noise_policy: str,
    random_seed: int,
) -> BaseClusteringResult:
    """Fit base clusters using k-means, HDBSCAN, or a hybrid strategy."""

    _validate_embeddings(embeddings)
    normalized_strategy = strategy.strip().lower()
    if normalized_strategy not in SUPPORTED_CLUSTER_STRATEGIES:
        allowed = ", ".join(sorted(SUPPORTED_CLUSTER_STRATEGIES))
        raise ClusteringError(
            f"Unsupported clustering_strategy '{strategy}'. Expected one of: {allowed}."
        )
    if hdbscan_min_cluster_size <= 1:
        raise ClusteringError(
            f"clustering_hdbscan_min_cluster_size must be > 1, got {hdbscan_min_cluster_size}."
        )
    if hdbscan_min_samples <= 0:
        raise ClusteringError(
            f"clustering_hdbscan_min_samples must be positive, got {hdbscan_min_samples}."
        )

    target = resolve_base_cluster_target(
        sample_count=embeddings.shape[0],
        requested_k=requested_k,
        leaf_mode=leaf_mode,
        target_leaf_size=target_leaf_size,
        min_leaf_clusters=min_leaf_clusters,
        max_leaf_clusters=max_leaf_clusters,
    )
    normalized_noise_policy = noise_policy.strip().lower()

    if normalized_strategy == "kmeans":
        labels, centroids, effective_k = fit_base_kmeans(
            embeddings,
            requested_k=target.requested_k,
            random_seed=random_seed,
        )
        silhouette_value, db_value, ch_value = _safe_clustering_scores(
            embeddings=embeddings,
            labels=labels,
            random_seed=random_seed,
        )
        return BaseClusteringResult(
            labels=labels,
            centroids=centroids,
            effective_k=effective_k,
            requested_k=target.requested_k,
            strategy=normalized_strategy,
            leaf_mode=target.leaf_mode,
            noise_policy=normalized_noise_policy,
            noise_count=0,
            raw_cluster_count=effective_k,
            refinement_splits=0,
            silhouette_score=silhouette_value,
            davies_bouldin_score=db_value,
            calinski_harabasz_score=ch_value,
            fallback_reason=None,
            auto_target_k=target.auto_target_k,
        )

    model = HDBSCAN(
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
        allow_single_cluster=True,
        copy=False,
    )
    raw_labels = model.fit_predict(embeddings).astype(int)
    raw_cluster_count = len({int(label) for label in raw_labels.tolist() if int(label) >= 0})
    noise_count = int(np.count_nonzero(raw_labels < 0))
    adjusted_labels, effective_noise_policy = _apply_noise_policy(
        embeddings=embeddings,
        labels=raw_labels,
        noise_policy=normalized_noise_policy,
    )
    fallback_reason: str | None = None
    refinement_splits = 0

    if normalized_strategy == "hybrid":
        adjusted_labels, refinement_splits = _refine_cluster_count_hybrid(
            embeddings=embeddings,
            labels=adjusted_labels,
            target_k=target.requested_k,
            random_seed=random_seed,
        )
        current_k = len({int(label) for label in adjusted_labels.tolist()})
        if current_k < min(2, target.requested_k):
            labels, centroids, effective_k = fit_base_kmeans(
                embeddings,
                requested_k=target.requested_k,
                random_seed=random_seed,
            )
            silhouette_value, db_value, ch_value = _safe_clustering_scores(
                embeddings=embeddings,
                labels=labels,
                random_seed=random_seed,
            )
            return BaseClusteringResult(
                labels=labels,
                centroids=centroids,
                effective_k=effective_k,
                requested_k=target.requested_k,
                strategy="hybrid",
                leaf_mode=target.leaf_mode,
                noise_policy=effective_noise_policy,
                noise_count=noise_count,
                raw_cluster_count=raw_cluster_count,
                refinement_splits=refinement_splits,
                silhouette_score=silhouette_value,
                davies_bouldin_score=db_value,
                calinski_harabasz_score=ch_value,
                fallback_reason="hdbscan_collapsed_to_single_cluster_fallback_to_kmeans",
                auto_target_k=target.auto_target_k,
            )

    normalized_labels = _normalize_labels(adjusted_labels)
    centroids = _centroids_from_labels(embeddings, normalized_labels)
    effective_k = centroids.shape[0]
    silhouette_value, db_value, ch_value = _safe_clustering_scores(
        embeddings=embeddings,
        labels=normalized_labels,
        random_seed=random_seed,
    )
    return BaseClusteringResult(
        labels=normalized_labels,
        centroids=centroids,
        effective_k=effective_k,
        requested_k=target.requested_k,
        strategy=normalized_strategy,
        leaf_mode=target.leaf_mode,
        noise_policy=effective_noise_policy,
        noise_count=noise_count,
        raw_cluster_count=raw_cluster_count,
        refinement_splits=refinement_splits,
        silhouette_score=silhouette_value,
        davies_bouldin_score=db_value,
        calinski_harabasz_score=ch_value,
        fallback_reason=fallback_reason,
        auto_target_k=target.auto_target_k,
    )


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

"""Tests for base clustering utilities."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from clio_pipeline.pipeline import fit_base_clusters, fit_base_kmeans, resolve_base_cluster_target
from clio_pipeline.pipeline.clustering import build_base_cluster_outputs
from clio_pipeline.schemas import Conversation, Facets, Message


def _conversation(conversation_id: str, user_id: str) -> Conversation:
    return Conversation(
        conversation_id=conversation_id,
        user_id=user_id,
        timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        messages=[
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi"),
        ],
    )


def _facet(conversation_id: str) -> Facets:
    return Facets(
        conversation_id=conversation_id,
        summary="General task support",
        task="Assist with task",
        language="English",
        turn_count=2,
        concerning_score=1,
    )


class TestFitBaseKMeans:
    def test_effective_k_is_capped_by_sample_count(self):
        embeddings = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [5.0, 5.0],
            ],
            dtype=float,
        )
        labels, centroids, effective_k = fit_base_kmeans(
            embeddings,
            requested_k=10,
            random_seed=42,
        )
        assert len(labels) == 3
        assert centroids.shape == (3, 2)
        assert effective_k == 3


class TestResolveBaseClusterTarget:
    def test_auto_mode_respects_bounds(self):
        resolved = resolve_base_cluster_target(
            sample_count=2000,
            requested_k=20,
            leaf_mode="auto",
            target_leaf_size=25,
            min_leaf_clusters=20,
            max_leaf_clusters=600,
        )
        assert resolved.requested_k == 80
        assert resolved.auto_target_k == 80

    def test_fixed_mode_uses_requested_k(self):
        resolved = resolve_base_cluster_target(
            sample_count=100,
            requested_k=20,
            leaf_mode="fixed",
            target_leaf_size=25,
            min_leaf_clusters=20,
            max_leaf_clusters=600,
        )
        assert resolved.requested_k == 20
        assert resolved.auto_target_k is None


class TestFitBaseClusters:
    def test_hybrid_strategy_splits_to_target_k(self):
        embeddings = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.0, 0.1],
                [0.1, 0.1],
                [3.0, 3.0],
                [3.1, 3.0],
                [3.0, 3.1],
                [3.1, 3.1],
            ],
            dtype=float,
        )
        result = fit_base_clusters(
            embeddings,
            strategy="hybrid",
            leaf_mode="fixed",
            requested_k=4,
            target_leaf_size=25,
            min_leaf_clusters=2,
            max_leaf_clusters=10,
            hdbscan_min_cluster_size=3,
            hdbscan_min_samples=1,
            noise_policy="nearest",
            random_seed=42,
        )
        assert result.effective_k == 4
        assert len(result.labels) == embeddings.shape[0]
        assert result.refinement_splits >= 0

    def test_hdbscan_strategy_handles_noise_with_singleton_policy(self):
        embeddings = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.2],
                [0.2, 0.0],
                [8.0, 8.0],  # outlier candidate
            ],
            dtype=float,
        )
        result = fit_base_clusters(
            embeddings,
            strategy="hdbscan",
            leaf_mode="fixed",
            requested_k=2,
            target_leaf_size=25,
            min_leaf_clusters=2,
            max_leaf_clusters=10,
            hdbscan_min_cluster_size=2,
            hdbscan_min_samples=1,
            noise_policy="singleton",
            random_seed=42,
        )
        assert result.effective_k >= 2
        assert np.all(result.labels >= 0)


class TestBuildBaseClusterOutputs:
    def test_threshold_flags_are_applied(self):
        conversations = [
            _conversation("conv-1", "user-1"),
            _conversation("conv-2", "user-2"),
            _conversation("conv-3", "user-2"),
        ]
        facets = [_facet("conv-1"), _facet("conv-2"), _facet("conv-3")]
        labels = np.array([0, 1, 1], dtype=int)

        clusters, assignments = build_base_cluster_outputs(
            conversations=conversations,
            facets=facets,
            labels=labels,
            min_unique_users=2,
            min_conversations_per_cluster=2,
        )

        by_cluster = {item["cluster_id"]: item for item in clusters}
        assert by_cluster[0]["kept_by_threshold"] is False
        assert by_cluster[1]["kept_by_threshold"] is False
        assert all("kept_by_threshold" in item for item in assignments)

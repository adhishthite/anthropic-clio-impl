"""Tests for hierarchy scaffold utilities."""

from __future__ import annotations

import numpy as np

from clio_pipeline.pipeline.hierarchy import build_multilevel_hierarchy_scaffold


class _FakeHierarchyClient:
    def complete_json(self, *, system_prompt: str, user_prompt: str, **kwargs) -> dict:
        return {
            "name": "Parent category",
            "description": "High-level grouping for related child clusters.",
        }


def test_build_multilevel_hierarchy_parallel_labeling():
    labeled_clusters = [
        {
            "cluster_id": 0,
            "name": "Cluster 0",
            "description": "Description 0",
            "size": 3,
            "unique_users": 3,
            "kept_by_threshold": True,
            "conversation_ids": ["c0"],
        },
        {
            "cluster_id": 1,
            "name": "Cluster 1",
            "description": "Description 1",
            "size": 3,
            "unique_users": 3,
            "kept_by_threshold": True,
            "conversation_ids": ["c1"],
        },
        {
            "cluster_id": 2,
            "name": "Cluster 2",
            "description": "Description 2",
            "size": 3,
            "unique_users": 3,
            "kept_by_threshold": True,
            "conversation_ids": ["c2"],
        },
        {
            "cluster_id": 3,
            "name": "Cluster 3",
            "description": "Description 3",
            "size": 3,
            "unique_users": 3,
            "kept_by_threshold": True,
            "conversation_ids": ["c3"],
        },
    ]
    leaf_embeddings = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.1, 0.2, 0.9],
            [4.0, 3.8, 1.0],
            [4.2, 4.1, 1.1],
        ],
        dtype=float,
    )

    hierarchy = build_multilevel_hierarchy_scaffold(
        labeled_clusters=labeled_clusters,
        leaf_embeddings=leaf_embeddings,
        llm_client=_FakeHierarchyClient(),
        max_levels=3,
        target_group_size=2,
        random_seed=7,
        max_label_concurrency=4,
    )

    assert hierarchy["leaf_cluster_count"] == len(labeled_clusters)
    assert hierarchy["top_level_cluster_count"] >= 1
    assert hierarchy["max_level"] >= 1
    assert len(hierarchy["nodes"]) >= len(labeled_clusters)

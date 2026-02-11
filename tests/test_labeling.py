"""Tests for cluster labeling utilities."""

from __future__ import annotations

from clio_pipeline.pipeline.labeling import label_clusters
from clio_pipeline.schemas import Facets


class _FakeLabelClient:
    def complete_json(self, *, system_prompt: str, user_prompt: str, **kwargs) -> dict:
        return {
            "name": "General support cluster",
            "description": "Users asked for general practical assistance.",
        }


def _facet(conversation_id: str, summary: str) -> Facets:
    return Facets(
        conversation_id=conversation_id,
        summary=summary,
        task="General support",
        language="English",
        language_confidence=0.9,
        turn_count=2,
        message_count=2,
        user_message_count=1,
        assistant_message_count=1,
        avg_user_message_length=20.0,
        avg_assistant_message_length=20.0,
        concerning_score=1,
    )


def test_label_clusters_parallel_mode_preserves_cluster_order():
    cluster_summaries = [
        {
            "cluster_id": 3,
            "size": 2,
            "unique_users": 2,
            "kept_by_threshold": True,
            "conversation_ids": ["conv-3-a", "conv-3-b"],
        },
        {
            "cluster_id": 1,
            "size": 2,
            "unique_users": 2,
            "kept_by_threshold": True,
            "conversation_ids": ["conv-1-a", "conv-1-b"],
        },
        {
            "cluster_id": 2,
            "size": 2,
            "unique_users": 2,
            "kept_by_threshold": True,
            "conversation_ids": ["conv-2-a", "conv-2-b"],
        },
    ]
    facets = [
        _facet("conv-1-a", "Account setup support"),
        _facet("conv-1-b", "Password reset support"),
        _facet("conv-2-a", "Bug troubleshooting support"),
        _facet("conv-2-b", "Error diagnosis support"),
        _facet("conv-3-a", "Writing assistance support"),
        _facet("conv-3-b", "Editing assistance support"),
    ]

    labeled = label_clusters(
        cluster_summaries=cluster_summaries,
        facets=facets,
        llm_client=_FakeLabelClient(),
        sample_size=2,
        max_concurrency=4,
    )

    assert len(labeled) == 3
    assert [item["cluster_id"] for item in labeled] == [1, 2, 3]
    assert all(item["labeling_fallback_used"] is False for item in labeled)

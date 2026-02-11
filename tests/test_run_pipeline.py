"""Tests for pipeline orchestration helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clio_pipeline.config import Settings
from clio_pipeline.mock_data import generate_mock_conversations
from clio_pipeline.pipeline import (
    build_run_fingerprint,
    initialize_run_artifacts,
    initialize_run_artifacts_streaming,
    load_phase2_facets,
    load_phase3_cluster_summaries,
    load_phase4_hierarchy,
    load_phase4_labeled_clusters,
    load_phase5_outputs,
    load_phase6_evaluation,
    run_phase1_dataset_load,
    run_phase2_facet_extraction,
    run_phase2_facet_extraction_streaming,
    run_phase3_base_clustering,
    run_phase4_cluster_labeling,
    run_phase4_hierarchy_scaffold,
    run_phase5_privacy_audit,
    run_phase6_evaluation,
)


class _FakeJsonClient:
    def complete_json(self, *, system_prompt: str, user_prompt: str, **kwargs) -> dict:
        return {
            "summary": "The user requested assistance with a task.",
            "task": "Assist with a general task",
            "language": "English",
            "concerning_score": 1,
        }


class _FakeLabelingClient:
    def complete_json(self, *, system_prompt: str, user_prompt: str, **kwargs) -> dict:
        if "higher-level category" in user_prompt.lower():
            return {
                "name": "Organize software and writing support",
                "description": "This parent cluster grouped related assistance themes.",
            }
        return {
            "name": "Assist with software and writing tasks",
            "description": "The cluster covered requests for practical task support.",
        }


class _FlakyLabelingClient:
    def __init__(self) -> None:
        self._calls = 0

    def complete_json(self, *, system_prompt: str, user_prompt: str, **kwargs) -> dict:
        self._calls += 1
        if self._calls % 2 == 0:
            raise RuntimeError("Synthetic labeling failure")
        if "higher-level category" in user_prompt.lower():
            return {
                "name": "Higher level fallback test group",
                "description": "Parent cluster label for test coverage.",
            }
        return {
            "name": "Stable generated label",
            "description": "Generated when synthetic error is not triggered.",
        }


class _FakePrivacyClient:
    def complete_json(self, *, system_prompt: str, user_prompt: str, **kwargs) -> dict:
        if "cluster_summary" in user_prompt:
            rating = 5
        elif "raw_conversation" in user_prompt:
            rating = 3
        else:
            rating = 4
        return {
            "rating": rating,
            "justification": "Content appears sufficiently general for this stage.",
        }


class _FakeEmbeddingClient:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [
            [float(len(text) % 7), float(index), 1.0, float((len(text) + index) % 5)]
            for index, text in enumerate(texts)
        ]


def _base_settings(tmp_path: Path) -> Settings:
    dataset_path = tmp_path / "data" / "mock" / "conversations_llm_200.jsonl"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    if not dataset_path.exists():
        rows = generate_mock_conversations(count=220, seed=7)
        dataset_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
    return Settings(
        openai_api_key="",
        azure_openai_api_key="",
        azure_openai_endpoint="",
        azure_openai_base_url="",
        openai_base_url="",
        jina_api_key="",
        input_conversations_path=dataset_path,
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "outputs",
        k_base_clusters=4,
        min_unique_users=1,
        min_conversations_per_cluster=1,
        hierarchy_top_k=2,
    )


class TestRunPipeline:
    def test_phase1_dataset_load(self, tmp_path: Path):
        settings = _base_settings(tmp_path)
        conversations, summary, dataset_path = run_phase1_dataset_load(settings)
        assert dataset_path.as_posix().endswith("data/mock/conversations_llm_200.jsonl")
        assert len(conversations) == summary.conversation_count
        assert summary.conversation_count >= 200

    def test_initialize_run_artifacts_creates_run_files(self, tmp_path: Path):
        settings = _base_settings(tmp_path)
        conversations, _, dataset_path = run_phase1_dataset_load(settings)
        run_id, run_root = initialize_run_artifacts(
            settings=settings,
            conversations=conversations,
            dataset_path=dataset_path,
        )

        assert len(run_id) == 12
        assert (run_root / "conversation.jsonl").exists()
        assert (run_root / "conversation.updated.jsonl").exists()
        assert (run_root / "run_manifest.json").exists()

    def test_initialize_run_artifacts_blocks_resume_fingerprint_drift(self, tmp_path: Path):
        settings = _base_settings(tmp_path)
        conversations, _, dataset_path = run_phase1_dataset_load(settings)
        run_fingerprint = build_run_fingerprint(
            settings,
            dataset_path=dataset_path,
            limit=5,
        )
        _, run_root = initialize_run_artifacts(
            settings=settings,
            conversations=conversations,
            dataset_path=dataset_path,
            run_id="run-test-fingerprint",
            run_fingerprint=run_fingerprint,
        )
        drifted_fingerprint = dict(run_fingerprint)
        drifted_fingerprint["facet_max_concurrency"] = 999

        with pytest.raises(ValueError, match="fingerprint mismatch"):
            initialize_run_artifacts(
                settings=settings,
                conversations=conversations,
                dataset_path=dataset_path,
                run_id="run-test-fingerprint",
                run_fingerprint=drifted_fingerprint,
                enforce_resume_fingerprint=True,
            )
        manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["run_fingerprint"]["facet_max_concurrency"] != 999

    def test_initialize_run_artifacts_streaming_creates_messages_only_files(
        self, tmp_path: Path
    ):
        settings = _base_settings(tmp_path)
        _, _, dataset_path = run_phase1_dataset_load(settings)
        run_id, run_root, summary = initialize_run_artifacts_streaming(
            settings=settings,
            dataset_path=dataset_path,
            chunk_size=5,
            limit=11,
            run_id="run-test-stream-init",
        )

        assert run_id == "run-test-stream-init"
        assert summary.conversation_count == 11
        assert (run_root / "conversation.jsonl").exists()
        assert (run_root / "conversation.updated.jsonl").exists()

        conversation_lines = (
            run_root / "conversation.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        updated_lines = (
            run_root / "conversation.updated.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        assert len(conversation_lines) == 11
        assert len(updated_lines) == 11

        manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["streaming_mode"] is True
        assert manifest["stream_chunk_size"] == 5

    def test_phase2_facet_extraction_saves_outputs(self, tmp_path: Path):
        settings = _base_settings(tmp_path)
        conversations, _, _ = run_phase1_dataset_load(settings)
        facets, run_root = run_phase2_facet_extraction(
            settings=settings,
            conversations=conversations,
            llm_client=_FakeJsonClient(),
            run_id="run-test-phase2",
            limit=5,
        )

        assert len(facets) == 5
        facets_path = run_root / "facets" / "facets.jsonl"
        facet_checkpoint_path = run_root / "facets" / "facet_checkpoint.json"
        manifest_path = run_root / "run_manifest.json"
        conversation_path = run_root / "conversation.jsonl"
        updated_conversation_path = run_root / "conversation.updated.jsonl"
        assert facets_path.exists()
        assert facet_checkpoint_path.exists()
        assert manifest_path.exists()
        assert conversation_path.exists()
        assert updated_conversation_path.exists()

        lines = facets_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 5

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["run_id"] == "run-test-phase2"
        assert manifest["conversation_count_processed"] == 5
        assert "facet_max_concurrency" in manifest
        assert "phase2_adaptive_concurrency_enabled" in manifest
        assert "facet_checkpoint_json" in manifest["output_files"]

    def test_phase2_facet_extraction_streaming_saves_outputs(self, tmp_path: Path):
        settings = _base_settings(tmp_path)
        _, _, dataset_path = run_phase1_dataset_load(settings)
        run_id, run_root, _ = initialize_run_artifacts_streaming(
            settings=settings,
            dataset_path=dataset_path,
            chunk_size=4,
            limit=10,
            run_id="run-test-phase2-stream",
        )

        facets, run_root = run_phase2_facet_extraction_streaming(
            settings=settings,
            dataset_path=dataset_path,
            run_id=run_id,
            stream_chunk_size=4,
            llm_client=_FakeJsonClient(),
            limit=10,
            total_conversations=10,
        )

        assert len(facets) == 10
        assert run_root.name == "run-test-phase2-stream"
        assert (run_root / "facets" / "facets.jsonl").exists()
        assert (run_root / "facets" / "facet_checkpoint.json").exists()
        updated_rows = [
            json.loads(line)
            for line in (run_root / "conversation.updated.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert len(updated_rows) == 10
        assert "analysis" in updated_rows[0]
        assert "facets" in updated_rows[0]["analysis"]

        manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["phase2_streaming_mode"] is True
        assert manifest["stream_chunk_size"] == 4

    def test_phase2_requires_key_without_injected_client(self, tmp_path: Path):
        settings = _base_settings(tmp_path)
        conversations, _, _ = run_phase1_dataset_load(settings)

        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            run_phase2_facet_extraction(
                settings=settings,
                conversations=conversations,
            )

    def test_phase3_base_clustering_saves_outputs(self, tmp_path: Path):
        settings = _base_settings(tmp_path)
        conversations, _, _ = run_phase1_dataset_load(settings)
        facets, run_root = run_phase2_facet_extraction(
            settings=settings,
            conversations=conversations,
            llm_client=_FakeJsonClient(),
            run_id="run-test-phase3",
            limit=6,
        )
        clusters, run_root = run_phase3_base_clustering(
            settings=settings,
            conversations=conversations,
            facets=facets,
            run_root=run_root,
            embedding_client=_FakeEmbeddingClient(),
        )

        assert len(clusters) >= 1
        assert (run_root / "embeddings" / "summary_embeddings.npy").exists()
        assert (run_root / "clusters" / "base_centroids.npy").exists()
        assert (run_root / "clusters" / "base_assignments.jsonl").exists()
        assert (run_root / "clusters" / "base_clusters.json").exists()
        assert (run_root / "viz" / "map_points.jsonl").exists()
        assert (run_root / "viz" / "map_clusters.json").exists()
        assert (run_root / "conversation.updated.jsonl").exists()

        manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
        assert "phase3_base_clustering" in manifest["completed_phases"]
        assert manifest["effective_k"] <= settings.k_base_clusters

    def test_phase3_requires_key_without_injected_client(self, tmp_path: Path):
        settings = _base_settings(tmp_path)
        conversations, _, _ = run_phase1_dataset_load(settings)
        facets, run_root = run_phase2_facet_extraction(
            settings=settings,
            conversations=conversations,
            llm_client=_FakeJsonClient(),
            run_id="run-test-phase3-key-check",
            limit=3,
        )

        with pytest.raises(ValueError, match="JINA_API_KEY"):
            run_phase3_base_clustering(
                settings=settings,
                conversations=conversations,
                facets=facets,
                run_root=run_root,
            )

    def test_phase4_labeling_saves_outputs(self, tmp_path: Path):
        settings = _base_settings(tmp_path)
        conversations, _, _ = run_phase1_dataset_load(settings)
        facets, run_root = run_phase2_facet_extraction(
            settings=settings,
            conversations=conversations,
            llm_client=_FakeJsonClient(),
            run_id="run-test-phase4-label",
            limit=6,
        )
        clusters, run_root = run_phase3_base_clustering(
            settings=settings,
            conversations=conversations,
            facets=facets,
            run_root=run_root,
            embedding_client=_FakeEmbeddingClient(),
        )
        labeled_clusters, run_root = run_phase4_cluster_labeling(
            settings=settings,
            facets=facets,
            cluster_summaries=clusters,
            run_root=run_root,
            conversations=conversations,
            llm_client=_FakeLabelingClient(),
        )

        assert len(labeled_clusters) >= 1
        labels_path = run_root / "clusters" / "labeled_clusters.json"
        assert labels_path.exists()
        assert (run_root / "clusters" / "cluster_label_checkpoint.json").exists()
        assert (run_root / "clusters" / "labeled_clusters.partial.jsonl").exists()

        manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
        assert "phase4_cluster_labeling" in manifest["completed_phases"]
        assert manifest["cluster_count_total"] == len(labeled_clusters)
        assert "cluster_label_max_concurrency" in manifest
        assert "phase4_cluster_adaptive_concurrency_enabled" in manifest
        assert "cluster_label_checkpoint_json" in manifest["output_files"]

    def test_phase4_labeling_handles_partial_failures(self, tmp_path: Path):
        settings = _base_settings(tmp_path)
        conversations, _, _ = run_phase1_dataset_load(settings)
        facets, run_root = run_phase2_facet_extraction(
            settings=settings,
            conversations=conversations,
            llm_client=_FakeJsonClient(),
            run_id="run-test-phase4-label-fallback",
            limit=6,
        )
        clusters, run_root = run_phase3_base_clustering(
            settings=settings,
            conversations=conversations,
            facets=facets,
            run_root=run_root,
            embedding_client=_FakeEmbeddingClient(),
        )
        labeled_clusters, run_root = run_phase4_cluster_labeling(
            settings=settings,
            facets=facets,
            cluster_summaries=clusters,
            run_root=run_root,
            conversations=conversations,
            llm_client=_FlakyLabelingClient(),
        )

        assert len(labeled_clusters) >= 1
        assert any(item["labeling_fallback_used"] for item in labeled_clusters)
        assert (run_root / "clusters" / "labeled_clusters_errors.jsonl").exists()

    def test_phase4_hierarchy_saves_outputs(self, tmp_path: Path):
        settings = _base_settings(tmp_path)
        conversations, _, _ = run_phase1_dataset_load(settings)
        facets, run_root = run_phase2_facet_extraction(
            settings=settings,
            conversations=conversations,
            llm_client=_FakeJsonClient(),
            run_id="run-test-phase4-hierarchy",
            limit=8,
        )
        clusters, run_root = run_phase3_base_clustering(
            settings=settings,
            conversations=conversations,
            facets=facets,
            run_root=run_root,
            embedding_client=_FakeEmbeddingClient(),
        )
        labeled_clusters, run_root = run_phase4_cluster_labeling(
            settings=settings,
            facets=facets,
            cluster_summaries=clusters,
            run_root=run_root,
            conversations=conversations,
            llm_client=_FakeLabelingClient(),
        )
        hierarchy, run_root = run_phase4_hierarchy_scaffold(
            settings=settings,
            labeled_clusters=labeled_clusters,
            run_root=run_root,
            llm_client=_FakeLabelingClient(),
            embedding_client=_FakeEmbeddingClient(),
        )

        assert hierarchy["leaf_cluster_count"] == len(labeled_clusters)
        assert hierarchy["top_level_cluster_count"] >= 1
        hierarchy_path = run_root / "clusters" / "hierarchy.json"
        assert hierarchy_path.exists()
        assert (run_root / "viz" / "tree_view.json").exists()
        assert (run_root / "clusters" / "hierarchy_checkpoint.json").exists()
        assert (run_root / "clusters" / "hierarchy_label_groups.partial.jsonl").exists()

        manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
        assert "phase4_hierarchy_scaffold" in manifest["completed_phases"]
        assert "hierarchy_label_max_concurrency" in manifest
        assert "phase4_hierarchy_adaptive_concurrency_enabled" in manifest
        assert "hierarchy_checkpoint_json" in manifest["output_files"]

    def test_phase5_privacy_audit_saves_outputs(self, tmp_path: Path):
        settings = _base_settings(tmp_path)
        conversations, _, _ = run_phase1_dataset_load(settings)
        facets, run_root = run_phase2_facet_extraction(
            settings=settings,
            conversations=conversations,
            llm_client=_FakeJsonClient(),
            run_id="run-test-phase5",
            limit=8,
        )
        clusters, run_root = run_phase3_base_clustering(
            settings=settings,
            conversations=conversations,
            facets=facets,
            run_root=run_root,
            embedding_client=_FakeEmbeddingClient(),
        )
        labeled_clusters, run_root = run_phase4_cluster_labeling(
            settings=settings,
            facets=facets,
            cluster_summaries=clusters,
            run_root=run_root,
            conversations=conversations,
            llm_client=_FakeLabelingClient(),
        )
        summary, gated_clusters, run_root = run_phase5_privacy_audit(
            settings=settings,
            conversations=conversations,
            facets=facets,
            labeled_clusters=labeled_clusters,
            run_root=run_root,
            llm_client=_FakePrivacyClient(),
            raw_sample_size=5,
        )

        assert "cluster_summary" in summary
        assert len(gated_clusters) == len(labeled_clusters)
        assert (run_root / "privacy" / "privacy_audit.json").exists()
        assert (run_root / "privacy" / "privacy_checkpoint.json").exists()
        assert (run_root / "clusters" / "labeled_clusters_privacy_filtered.json").exists()
        assert (run_root / "conversation.updated.jsonl").exists()

        manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
        assert "phase5_privacy_audit" in manifest["completed_phases"]
        assert "privacy_max_concurrency" in manifest
        assert "phase5_adaptive_concurrency_enabled" in manifest
        assert "privacy_checkpoint_json" in manifest["output_files"]

    def test_phase6_evaluation_saves_outputs(self, tmp_path: Path):
        settings = _base_settings(tmp_path)
        conversations, _, dataset_path = run_phase1_dataset_load(settings)
        _, run_root = initialize_run_artifacts(
            settings=settings,
            conversations=conversations[:5],
            dataset_path=dataset_path,
            run_id="run-test-phase6",
        )
        results, run_root = run_phase6_evaluation(
            settings=settings,
            run_root=run_root,
            count=20,
            topic_count=4,
            language_count=3,
            seed=7,
        )

        assert results["synthetic_count"] == 20
        assert "privacy_summary" in results["ablations"]
        assert (run_root / "eval" / "phase6_metrics.json").exists()
        assert (run_root / "eval" / "synthetic_conversations.jsonl").exists()
        assert (run_root / "eval" / "report.md").exists()

        manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
        assert "phase6_evaluation" in manifest["completed_phases"]

    def test_resume_loaders_return_existing_artifacts(self, tmp_path: Path):
        settings = _base_settings(tmp_path)
        conversations, _, _ = run_phase1_dataset_load(settings)
        facets, run_root = run_phase2_facet_extraction(
            settings=settings,
            conversations=conversations,
            llm_client=_FakeJsonClient(),
            run_id="run-test-resume",
            limit=6,
        )
        clusters, run_root = run_phase3_base_clustering(
            settings=settings,
            conversations=conversations,
            facets=facets,
            run_root=run_root,
            embedding_client=_FakeEmbeddingClient(),
        )
        labeled_clusters, run_root = run_phase4_cluster_labeling(
            settings=settings,
            facets=facets,
            cluster_summaries=clusters,
            run_root=run_root,
            conversations=conversations,
            llm_client=_FakeLabelingClient(),
        )
        run_phase4_hierarchy_scaffold(
            settings=settings,
            labeled_clusters=labeled_clusters,
            run_root=run_root,
            llm_client=_FakeLabelingClient(),
            embedding_client=_FakeEmbeddingClient(),
        )
        run_phase5_privacy_audit(
            settings=settings,
            conversations=conversations,
            facets=facets,
            labeled_clusters=labeled_clusters,
            run_root=run_root,
            llm_client=_FakePrivacyClient(),
            raw_sample_size=5,
        )
        run_phase6_evaluation(
            settings=settings,
            run_root=run_root,
            count=20,
            topic_count=4,
            language_count=3,
            seed=7,
        )

        assert len(load_phase2_facets(run_root)) == len(facets)
        assert len(load_phase3_cluster_summaries(run_root)) == len(clusters)
        assert len(load_phase4_labeled_clusters(run_root)) == len(labeled_clusters)
        assert load_phase4_hierarchy(run_root)["leaf_cluster_count"] == len(labeled_clusters)
        summary, gated = load_phase5_outputs(run_root)
        assert "cluster_summary" in summary
        assert len(gated) == len(labeled_clusters)
        assert load_phase6_evaluation(run_root)["synthetic_count"] == 20

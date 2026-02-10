"""Tests for synthetic evaluation utilities."""

from __future__ import annotations

from pathlib import Path

from clio_pipeline.config import Settings
from clio_pipeline.pipeline.evaluate import (
    generate_synthetic_eval_conversations,
    run_synthetic_evaluation_suite,
)


def test_generate_synthetic_eval_conversations_shape():
    conversations = generate_synthetic_eval_conversations(
        count=20,
        topic_count=4,
        language_count=3,
        seed=11,
    )
    assert len(conversations) == 20
    assert conversations[0].metadata["source"] == "synthetic_eval"
    assert "ground_truth_topic" in conversations[0].metadata
    assert "ground_truth_language" in conversations[0].metadata


def test_run_synthetic_evaluation_suite_returns_metrics(tmp_path: Path):
    settings = Settings(
        openai_api_key="",
        azure_openai_api_key="",
        azure_openai_endpoint="",
        azure_openai_base_url="",
        openai_base_url="",
        jina_api_key="",
        output_dir=tmp_path / "runs",
    )
    results = run_synthetic_evaluation_suite(
        settings=settings,
        count=20,
        topic_count=4,
        language_count=3,
        seed=5,
    )
    assert results["synthetic_count"] == 20
    assert "privacy_summary" in results["ablations"]
    assert "raw_user_text" in results["ablations"]
    assert 0.0 <= results["ablations"]["privacy_summary"]["accuracy"] <= 1.0
    assert len(results["synthetic_conversations"]) == 20

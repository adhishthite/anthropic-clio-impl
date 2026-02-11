"""Tests for privacy audit utilities."""

from __future__ import annotations

from clio_pipeline.pipeline.privacy_audit import (
    apply_cluster_privacy_gate,
    audit_content_batch,
    run_privacy_auditor_validation,
    summarize_privacy_records,
)


def test_summarize_privacy_records_counts():
    records = [
        {"rating": 2},
        {"rating": 3},
        {"rating": 4},
        {"rating": 5},
    ]
    summary = summarize_privacy_records(records, threshold=3)
    assert summary["total"] == 4
    assert summary["pass_count"] == 3
    assert summary["fail_count"] == 1
    assert summary["rating_counts"]["2"] == 1


def test_apply_cluster_privacy_gate():
    labeled_clusters = [
        {"cluster_id": 1, "name": "A", "description": "B", "kept_by_threshold": True},
        {"cluster_id": 2, "name": "C", "description": "D", "kept_by_threshold": False},
    ]
    audits = [
        {"cluster_id": 1, "rating": 4, "justification": "general"},
        {"cluster_id": 2, "rating": 5, "justification": "general"},
    ]
    output = apply_cluster_privacy_gate(
        labeled_clusters=labeled_clusters,
        cluster_audits=audits,
        threshold=3,
    )
    by_id = {item["cluster_id"]: item for item in output}
    assert by_id[1]["kept_by_privacy"] is True
    assert by_id[1]["final_kept"] is True
    assert by_id[2]["kept_by_privacy"] is True
    assert by_id[2]["final_kept"] is False


class _FakePrivacyClient:
    def complete_json(self, *, system_prompt: str, user_prompt: str, **kwargs) -> dict:
        if "Jane Doe" in user_prompt or "Northbridge Biotech" in user_prompt:
            rating = 1
        elif "single startup" in user_prompt:
            rating = 2
        else:
            rating = 5
        return {
            "rating": rating,
            "justification": "Synthetic validator response.",
        }


class _ErrorPrivacyClient:
    def complete_json(self, *, system_prompt: str, user_prompt: str, **kwargs) -> dict:
        raise RuntimeError("Synthetic audit failure")


class _BatchPrivacyClient:
    def complete_json(self, *, system_prompt: str, user_prompt: str, **kwargs) -> dict:
        if "content_id: item-1" in user_prompt and "content_id: item-2" in user_prompt:
            return {
                "audits": [
                    {"content_id": "item-1", "rating": 4, "justification": "Generalized content."},
                    {"content_id": "item-2", "rating": 2, "justification": "Narrow scope details."},
                ]
            }
        return {"audits": [{"content_id": "item-1", "rating": 4, "justification": "Generalized."}]}


def test_run_privacy_auditor_validation():
    result = run_privacy_auditor_validation(_FakePrivacyClient())
    assert result["total_cases"] >= 1
    assert 0.0 <= result["in_range_rate"] <= 1.0
    assert len(result["records"]) == result["total_cases"]


def test_run_privacy_auditor_validation_fallback_on_errors():
    result = run_privacy_auditor_validation(_ErrorPrivacyClient())
    assert result["total_cases"] >= 1
    assert all(item["audit_fallback_used"] is True for item in result["records"])


def test_audit_content_batch_success():
    results, errors = audit_content_batch(
        stage="facet_summary",
        items=[("item-1", "summary=a"), ("item-2", "summary=b")],
        llm_client=_BatchPrivacyClient(),
    )
    assert not errors
    assert results["item-1"]["rating"] == 4
    assert results["item-2"]["rating"] == 2


def test_audit_content_batch_reports_missing_ids():
    results, errors = audit_content_batch(
        stage="facet_summary",
        items=[("item-1", "summary=a"), ("item-2", "summary=b"), ("item-3", "summary=c")],
        llm_client=_BatchPrivacyClient(),
    )
    assert "item-1" in results
    missing = [item for item in errors if item["error_type"] == "MissingContentIdInBatchOutput"]
    assert any(item["content_id"] == "item-3" for item in missing)

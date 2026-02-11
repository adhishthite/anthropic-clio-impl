"""Privacy auditing and gating logic for Phase 5."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

from clio_pipeline.models import LLMJsonClient
from clio_pipeline.prompts import PRIVACY_AUDIT_SYSTEM_PROMPT, build_privacy_audit_user_prompt
from clio_pipeline.schemas import Conversation, Facets


class PrivacyAuditError(ValueError):
    """Raised when privacy audit responses fail validation."""


class _PrivacyAuditPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rating: int = Field(ge=1, le=5)
    justification: str = Field(min_length=1, max_length=1000)


class _PrivacyAuditBatchItemPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content_id: str = Field(min_length=1)
    rating: int = Field(ge=1, le=5)
    justification: str = Field(min_length=1, max_length=1000)


class _PrivacyAuditBatchPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audits: list[_PrivacyAuditBatchItemPayload]


_PRIVACY_AUDIT_BATCH_SYSTEM_PROMPT = """You are a privacy auditor for
aggregated AI-usage analytics outputs.

Evaluate each content item on a 1-5 privacy-preservation scale:
1 = Not privacy-preserving (identifiable individual or very small group)
2 = Slightly privacy-preserving (identifiable specific organization/group)
3 = Fairly privacy-preserving (could narrow to a few thousand)
4 = Mostly privacy-preserving (general, needs substantial context to identify)
5 = Fully privacy-preserving (general, no identifying details)

Return strict JSON with exactly this shape:
{
  "audits": [
    {
      "content_id": "<copied exactly from input>",
      "rating": <integer 1-5>,
      "justification": "<brief rationale>"
    }
  ]
}

Rules:
- Include exactly one audit item per input content_id.
- Do not omit or invent content_ids.
- Preserve content_id values exactly.
"""


_PRIVACY_VALIDATION_SET: list[dict] = [
    {
        "case_id": "pv-001",
        "stage": "validation",
        "content": "Jane Doe at Acme Corp requested salary details for employee 412.",
        "expected_min_rating": 1,
        "expected_max_rating": 2,
    },
    {
        "case_id": "pv-002",
        "stage": "validation",
        "content": "A single startup in Austin asked for legal tax advice in a private context.",
        "expected_min_rating": 2,
        "expected_max_rating": 3,
    },
    {
        "case_id": "pv-003",
        "stage": "validation",
        "content": "Users asked for guidance on debugging Python web APIs.",
        "expected_min_rating": 4,
        "expected_max_rating": 5,
    },
    {
        "case_id": "pv-004",
        "stage": "validation",
        "content": "Many conversations involve drafting marketing copy for product launches.",
        "expected_min_rating": 4,
        "expected_max_rating": 5,
    },
    {
        "case_id": "pv-005",
        "stage": "validation",
        "content": (
            "An employee at Northbridge Biotech asked for merger-sensitive negotiation tactics."
        ),
        "expected_min_rating": 1,
        "expected_max_rating": 2,
    },
    {
        "case_id": "pv-006",
        "stage": "validation",
        "content": "Conversations discuss general language translation tasks across teams.",
        "expected_min_rating": 4,
        "expected_max_rating": 5,
    },
]


def _audit_content(*, stage: str, content: str, llm_client: LLMJsonClient) -> dict:
    try:
        payload = llm_client.complete_json(
            system_prompt=PRIVACY_AUDIT_SYSTEM_PROMPT,
            user_prompt=build_privacy_audit_user_prompt(stage=stage, content=content),
            schema_name="privacy_audit_payload",
            json_schema=_PrivacyAuditPayload.model_json_schema(),
            strict_schema=True,
        )
        parsed = _PrivacyAuditPayload.model_validate(payload)
    except Exception as exc:
        return {
            "rating": 1,
            "justification": (
                f"Fallback privacy rating due to audit error at stage '{stage}'. "
                "Conservatively rated as 1."
            ),
            "fallback_used": True,
            "error": str(exc),
        }

    return {
        "rating": int(parsed.rating),
        "justification": parsed.justification.strip(),
        "fallback_used": False,
        "error": None,
    }


def audit_content(*, stage: str, content: str, llm_client: LLMJsonClient) -> dict:
    """Audit one text block and return normalized rating payload."""

    return _audit_content(stage=stage, content=content, llm_client=llm_client)


def _build_privacy_audit_batch_user_prompt(*, stage: str, items: list[tuple[str, str]]) -> str:
    """Build prompt for privacy auditing a batch of text blocks."""

    sections: list[str] = []
    for index, (content_id, content) in enumerate(items, start=1):
        sections.append(
            f"Item {index}\n"
            f"content_id: {content_id}\n"
            "<content>\n"
            f"{content}\n"
            "</content>\n"
        )

    return (
        "Assess privacy preservation for each content item.\n"
        f"stage: {stage}\n\n"
        + "\n---\n".join(sections)
    )


def audit_content_batch(
    *,
    stage: str,
    items: list[tuple[str, str]],
    llm_client: LLMJsonClient,
) -> tuple[dict[str, dict], list[dict]]:
    """Audit multiple text blocks in a single LLM call."""

    if not items:
        return {}, []

    try:
        payload = llm_client.complete_json(
            system_prompt=_PRIVACY_AUDIT_BATCH_SYSTEM_PROMPT,
            user_prompt=_build_privacy_audit_batch_user_prompt(stage=stage, items=items),
            schema_name="privacy_audit_batch_payload",
            json_schema=_PrivacyAuditBatchPayload.model_json_schema(),
            strict_schema=True,
        )
        parsed = _PrivacyAuditBatchPayload.model_validate(payload)
    except Exception as exc:
        raise PrivacyAuditError(f"Privacy batch payload failed validation: {exc}") from exc

    expected_ids = {content_id for content_id, _ in items}
    seen_ids: set[str] = set()
    results_by_id: dict[str, dict] = {}
    errors: list[dict] = []

    for item in parsed.audits:
        content_id = item.content_id
        if content_id in seen_ids:
            errors.append(
                {
                    "content_id": content_id,
                    "error_type": "DuplicateContentIdInBatchOutput",
                    "error": "Batch output repeated content_id.",
                }
            )
            continue
        if content_id not in expected_ids:
            errors.append(
                {
                    "content_id": content_id,
                    "error_type": "UnexpectedContentIdInBatchOutput",
                    "error": "Batch output contained unknown content_id.",
                }
            )
            continue

        seen_ids.add(content_id)
        results_by_id[content_id] = {
            "rating": int(item.rating),
            "justification": item.justification.strip(),
            "fallback_used": False,
            "error": None,
        }

    for content_id, _ in items:
        if content_id not in seen_ids:
            errors.append(
                {
                    "content_id": content_id,
                    "error_type": "MissingContentIdInBatchOutput",
                    "error": "Batch output omitted this content_id.",
                }
            )

    return results_by_id, errors


def run_privacy_auditor_validation(
    llm_client: LLMJsonClient,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    """Evaluate auditor consistency on a fixed validation set."""

    records: list[dict] = []
    in_range_count = 0
    absolute_errors: list[int] = []

    for index, case in enumerate(_PRIVACY_VALIDATION_SET, start=1):
        result = _audit_content(
            stage=case["stage"],
            content=case["content"],
            llm_client=llm_client,
        )
        rating = int(result["rating"])
        lower = int(case["expected_min_rating"])
        upper = int(case["expected_max_rating"])
        in_range = lower <= rating <= upper
        if in_range:
            in_range_count += 1

        midpoint = round((lower + upper) / 2)
        absolute_errors.append(abs(rating - midpoint))
        records.append(
            {
                "case_id": case["case_id"],
                "rating": rating,
                "expected_min_rating": lower,
                "expected_max_rating": upper,
                "in_expected_range": in_range,
                "justification": result["justification"],
                "audit_fallback_used": bool(result.get("fallback_used", False)),
                "audit_error": result.get("error"),
            }
        )
        if progress_callback is not None:
            progress_callback(index, len(_PRIVACY_VALIDATION_SET))

    total = len(records)
    mean_absolute_error = (
        float(sum(absolute_errors) / len(absolute_errors)) if absolute_errors else 0.0
    )
    return {
        "total_cases": total,
        "in_range_count": in_range_count,
        "in_range_rate": (in_range_count / total) if total else 0.0,
        "mean_absolute_error": mean_absolute_error,
        "records": records,
    }


def audit_raw_conversations(
    *,
    conversations: list[Conversation],
    llm_client: LLMJsonClient,
    sample_limit: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Audit a sample of raw conversations for baseline privacy score."""

    if sample_limit <= 0:
        raise PrivacyAuditError(f"sample_limit must be positive, got {sample_limit}.")

    records: list[dict] = []
    sampled = conversations[:sample_limit]
    for index, conversation in enumerate(sampled, start=1):
        transcript = "\n".join(
            f"{message.role.upper()}: {message.content}" for message in conversation.messages
        )
        result = _audit_content(
            stage="raw_conversation",
            content=transcript,
            llm_client=llm_client,
        )
        records.append(
            {
                "stage": "raw_conversation",
                "conversation_id": conversation.conversation_id,
                "rating": result["rating"],
                "justification": result["justification"],
                "audit_fallback_used": bool(result.get("fallback_used", False)),
                "audit_error": result.get("error"),
            }
        )
        if progress_callback is not None:
            progress_callback(index, len(sampled))
    return records


def audit_facets(
    *,
    facets: list[Facets],
    llm_client: LLMJsonClient,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Audit extracted facets privacy."""

    records: list[dict] = []
    for index, facet in enumerate(facets, start=1):
        content = (
            f"summary={facet.summary}\n"
            f"task={facet.task}\n"
            f"language={facet.language}\n"
            f"turn_count={facet.turn_count}"
        )
        result = _audit_content(
            stage="facet_summary",
            content=content,
            llm_client=llm_client,
        )
        records.append(
            {
                "stage": "facet_summary",
                "conversation_id": facet.conversation_id,
                "rating": result["rating"],
                "justification": result["justification"],
                "audit_fallback_used": bool(result.get("fallback_used", False)),
                "audit_error": result.get("error"),
            }
        )
        if progress_callback is not None:
            progress_callback(index, len(facets))
    return records


def audit_labeled_clusters(
    *,
    labeled_clusters: list[dict],
    llm_client: LLMJsonClient,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Audit labeled cluster outputs for privacy."""

    records: list[dict] = []
    for index, cluster in enumerate(labeled_clusters, start=1):
        content = f"{cluster['name']}: {cluster['description']}"
        result = _audit_content(
            stage="cluster_summary",
            content=content,
            llm_client=llm_client,
        )
        records.append(
            {
                "stage": "cluster_summary",
                "cluster_id": int(cluster["cluster_id"]),
                "rating": result["rating"],
                "justification": result["justification"],
                "audit_fallback_used": bool(result.get("fallback_used", False)),
                "audit_error": result.get("error"),
            }
        )
        if progress_callback is not None:
            progress_callback(index, len(labeled_clusters))
    return records


def summarize_privacy_records(records: list[dict], *, threshold: int) -> dict:
    """Compute aggregate statistics from privacy audit records."""

    ratings = [int(item["rating"]) for item in records]
    counts = Counter(ratings)
    total = len(ratings)
    pass_count = sum(1 for rating in ratings if rating >= threshold)
    fail_count = total - pass_count
    return {
        "total": total,
        "threshold": threshold,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_rate": (pass_count / total) if total else 0.0,
        "rating_counts": {str(key): counts[key] for key in sorted(counts)},
    }


def apply_cluster_privacy_gate(
    *,
    labeled_clusters: list[dict],
    cluster_audits: list[dict],
    threshold: int,
) -> list[dict]:
    """Attach privacy ratings to clusters and apply keep/drop gate."""

    by_cluster_id = {int(item["cluster_id"]): item for item in cluster_audits}
    output: list[dict] = []

    for cluster in labeled_clusters:
        cluster_id = int(cluster["cluster_id"])
        audit = by_cluster_id.get(cluster_id)
        rating = int(audit["rating"]) if audit else 1
        justification = str(audit["justification"]) if audit else "Missing privacy audit result."
        kept_by_privacy = rating >= threshold
        kept_by_threshold = bool(cluster.get("kept_by_threshold", False))

        updated = dict(cluster)
        updated["privacy_rating"] = rating
        updated["privacy_justification"] = justification
        updated["kept_by_privacy"] = kept_by_privacy
        updated["final_kept"] = bool(kept_by_privacy and kept_by_threshold)
        output.append(updated)

    return output

"""Prompts for Phase 5 privacy auditing."""

from __future__ import annotations

PRIVACY_AUDIT_SYSTEM_PROMPT = """You are a privacy auditor for
aggregated AI-usage analytics outputs.

Evaluate whether provided text is privacy-preserving on a 1-5 scale:
1 = Not privacy-preserving (identifiable individual or very small group)
2 = Slightly privacy-preserving (identifiable specific organization/group)
3 = Fairly privacy-preserving (could narrow to a few thousand)
4 = Mostly privacy-preserving (general, needs substantial context to identify)
5 = Fully privacy-preserving (general, no identifying details)

Return strict JSON with exactly:
{
  "rating": <integer 1-5>,
  "justification": "<brief rationale>"
}
"""


def build_privacy_audit_user_prompt(*, stage: str, content: str) -> str:
    """Build prompt for privacy auditing one text block."""

    return (
        "Assess privacy preservation for this content.\n"
        f"stage: {stage}\n"
        "<content>\n"
        f"{content}\n"
        "</content>\n"
    )

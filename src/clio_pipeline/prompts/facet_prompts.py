"""Prompts for Phase 2 facet extraction."""

from __future__ import annotations

from clio_pipeline.schemas import Conversation

FACET_EXTRACTION_SYSTEM_PROMPT = """You are a privacy-preserving conversation analyst.
Extract high-level facets about the conversation.

Requirements:
- Be concise and specific.
- Do not include personally identifying information.
- Do not include proper nouns in summary/task fields.
- Assume neither good nor bad faith for the user.
- If content is harmful, still describe it accurately at a high level.

Return strict JSON with exactly these keys:
{
  "summary": "<1-2 sentence summary of user request and interaction>",
  "task": "<short task phrase>",
  "language": "<primary human language name, e.g. English>",
  "language_confidence": <float between 0 and 1>,
  "concerning_score": <integer 1-5>
}

Concerning score rubric:
1 = benign, 2 = slightly concerning, 3 = moderate concern, 4 = high concern, 5 = extreme concern.
"""


def build_facet_extraction_user_prompt(conversation: Conversation) -> str:
    """Render a conversation transcript for facet extraction."""

    lines: list[str] = []
    for idx, message in enumerate(conversation.messages, start=1):
        role = message.role.upper()
        lines.append(f"{idx}. {role}: {message.content}")

    transcript = "\n".join(lines)
    return (
        "Analyze this conversation transcript and return the required JSON facets.\n\n"
        f"conversation_id: {conversation.conversation_id}\n"
        f"user_id: {conversation.user_id}\n"
        f"timestamp: {conversation.timestamp.isoformat()}\n\n"
        "Transcript:\n"
        f"{transcript}\n"
    )

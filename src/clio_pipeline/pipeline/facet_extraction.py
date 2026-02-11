"""Phase 2 facet extraction stage."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from clio_pipeline.models import LLMJsonClient
from clio_pipeline.prompts import (
    FACET_EXTRACTION_SYSTEM_PROMPT,
    build_facet_extraction_user_prompt,
)
from clio_pipeline.schemas import Conversation, Facets


class FacetExtractionError(ValueError):
    """Raised when facet extraction fails validation."""


class _FacetPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1)
    task: str = Field(min_length=1)
    language: str = Field(min_length=1)
    language_confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    concerning_score: int = Field(ge=1, le=5)


class _FacetBatchItemPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    task: str = Field(min_length=1)
    language: str = Field(min_length=1)
    language_confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    concerning_score: int = Field(ge=1, le=5)


class _FacetBatchPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facets: list[_FacetBatchItemPayload]


_FACET_BATCH_SYSTEM_PROMPT = """You are a privacy-preserving conversation analyst.
Extract high-level facets for each input conversation.

Requirements:
- Be concise and specific.
- Do not include personally identifying information.
- Do not include proper nouns in summary/task fields.
- Assume neither good nor bad faith for the user.
- If content is harmful, still describe it accurately at a high level.

Return strict JSON with exactly this shape:
{
  "facets": [
    {
      "conversation_id": "<copied exactly from input>",
      "summary": "<1-2 sentence summary of user request and interaction>",
      "task": "<short task phrase>",
      "language": "<primary human language name, e.g. English>",
      "language_confidence": <float between 0 and 1>,
      "concerning_score": <integer 1-5>
    }
  ]
}

Rules:
- Include exactly one facet item per input conversation_id.
- Do not omit any conversation_id and do not add extra conversation_ids.
- Keep `conversation_id` values unchanged.
"""


def _build_facet_batch_user_prompt(conversations: list[Conversation]) -> str:
    """Render a batch transcript prompt for facet extraction."""

    sections: list[str] = []
    for index, conversation in enumerate(conversations, start=1):
        lines = [
            f"{turn_index}. {message.role.upper()}: {message.content}"
            for turn_index, message in enumerate(conversation.messages, start=1)
        ]
        transcript = "\n".join(lines)
        sections.append(
            f"Conversation {index}\n"
            f"conversation_id: {conversation.conversation_id}\n"
            f"user_id: {conversation.user_id}\n"
            f"timestamp: {conversation.timestamp.isoformat()}\n"
            "Transcript:\n"
            f"{transcript}\n"
        )

    body = "\n---\n".join(sections)
    return (
        "Analyze each conversation transcript and return one facet per conversation_id.\n\n"
        f"{body}"
    )


def _build_facet_output(
    *,
    conversation: Conversation,
    summary: str,
    task: str,
    language: str,
    language_confidence: float,
    concerning_score: int,
) -> Facets:
    """Build a Facets object with deterministic conversation stats."""

    user_messages = [message.content for message in conversation.messages if message.role == "user"]
    assistant_messages = [
        message.content for message in conversation.messages if message.role == "assistant"
    ]
    message_count = len(conversation.messages)
    user_count = len(user_messages)
    assistant_count = len(assistant_messages)
    avg_user_len = (
        sum(len(message) for message in user_messages) / user_count if user_count else 0.0
    )
    avg_assistant_len = (
        sum(len(message) for message in assistant_messages) / assistant_count
        if assistant_count
        else 0.0
    )

    return Facets(
        conversation_id=conversation.conversation_id,
        summary=summary.strip(),
        task=task.strip(),
        language=language.strip(),
        language_confidence=language_confidence,
        turn_count=len(conversation.messages),
        message_count=message_count,
        user_message_count=user_count,
        assistant_message_count=assistant_count,
        avg_user_message_length=avg_user_len,
        avg_assistant_message_length=avg_assistant_len,
        concerning_score=concerning_score,
    )


def extract_conversation_facets(
    conversation: Conversation,
    llm_client: LLMJsonClient,
) -> Facets:
    """Extract normalized facets for one conversation."""

    payload = llm_client.complete_json(
        system_prompt=FACET_EXTRACTION_SYSTEM_PROMPT,
        user_prompt=build_facet_extraction_user_prompt(conversation),
        schema_name="facet_payload",
        json_schema=_FacetPayload.model_json_schema(),
        strict_schema=True,
    )

    try:
        parsed = _FacetPayload.model_validate(payload)
    except Exception as exc:
        raise FacetExtractionError(
            f"Facet payload failed validation for conversation "
            f"'{conversation.conversation_id}': {exc}"
        ) from exc

    return _build_facet_output(
        conversation=conversation,
        summary=parsed.summary,
        task=parsed.task,
        language=parsed.language,
        language_confidence=parsed.language_confidence,
        concerning_score=parsed.concerning_score,
    )


def extract_facets_for_conversation_batch(
    conversations: list[Conversation],
    llm_client: LLMJsonClient,
) -> tuple[list[Facets], list[dict]]:
    """Extract facets for a batch of conversations in one LLM call."""

    if not conversations:
        return [], []

    payload = llm_client.complete_json(
        system_prompt=_FACET_BATCH_SYSTEM_PROMPT,
        user_prompt=_build_facet_batch_user_prompt(conversations),
        schema_name="facet_batch_payload",
        json_schema=_FacetBatchPayload.model_json_schema(),
        strict_schema=True,
    )
    try:
        parsed = _FacetBatchPayload.model_validate(payload)
    except Exception as exc:
        raise FacetExtractionError(f"Facet batch payload failed validation: {exc}") from exc

    conversation_by_id = {
        conversation.conversation_id: conversation for conversation in conversations
    }
    seen_ids: set[str] = set()
    facets: list[Facets] = []
    errors: list[dict] = []

    for item in parsed.facets:
        conversation_id = item.conversation_id
        if conversation_id in seen_ids:
            errors.append(
                {
                    "conversation_id": conversation_id,
                    "error_type": "DuplicateConversationIdInBatchOutput",
                    "error": "Batch output repeated conversation_id.",
                }
            )
            continue

        conversation = conversation_by_id.get(conversation_id)
        if conversation is None:
            errors.append(
                {
                    "conversation_id": conversation_id,
                    "error_type": "UnexpectedConversationIdInBatchOutput",
                    "error": "Batch output contained unknown conversation_id.",
                }
            )
            continue

        seen_ids.add(conversation_id)
        facets.append(
            _build_facet_output(
                conversation=conversation,
                summary=item.summary,
                task=item.task,
                language=item.language,
                language_confidence=item.language_confidence,
                concerning_score=item.concerning_score,
            )
        )

    for conversation in conversations:
        if conversation.conversation_id not in seen_ids:
            errors.append(
                {
                    "conversation_id": conversation.conversation_id,
                    "error_type": "MissingConversationInBatchOutput",
                    "error": "Batch output omitted this conversation_id.",
                }
            )

    return facets, errors


def extract_facets_batch(
    conversations: list[Conversation],
    llm_client: LLMJsonClient,
) -> list[Facets]:
    """Extract facets for a list of conversations."""

    return [extract_conversation_facets(conversation, llm_client) for conversation in conversations]

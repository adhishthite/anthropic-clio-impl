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
        summary=parsed.summary.strip(),
        task=parsed.task.strip(),
        language=parsed.language.strip(),
        language_confidence=parsed.language_confidence,
        turn_count=len(conversation.messages),
        message_count=message_count,
        user_message_count=user_count,
        assistant_message_count=assistant_count,
        avg_user_message_length=avg_user_len,
        avg_assistant_message_length=avg_assistant_len,
        concerning_score=parsed.concerning_score,
    )


def extract_facets_batch(
    conversations: list[Conversation],
    llm_client: LLMJsonClient,
) -> list[Facets]:
    """Extract facets for a list of conversations."""

    return [extract_conversation_facets(conversation, llm_client) for conversation in conversations]

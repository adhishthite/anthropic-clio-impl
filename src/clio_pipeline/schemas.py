"""Core data schemas for the CLIO pipeline."""

from datetime import datetime

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single message in a conversation."""

    role: str
    content: str


class Conversation(BaseModel):
    """A raw conversation record."""

    conversation_id: str
    user_id: str
    timestamp: datetime
    messages: list[Message]
    metadata: dict = Field(default_factory=dict)


class Facets(BaseModel):
    """Extracted facets for a single conversation."""

    conversation_id: str
    summary: str
    task: str
    language: str
    language_confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    turn_count: int
    message_count: int = 0
    user_message_count: int = 0
    assistant_message_count: int = 0
    avg_user_message_length: float = 0.0
    avg_assistant_message_length: float = 0.0
    concerning_score: int = Field(default=1, ge=1, le=5)


class ClusterInfo(BaseModel):
    """Metadata for a single cluster."""

    cluster_id: int
    name: str
    description: str
    size: int
    unique_users: int
    privacy_score: float | None = None
    parent_cluster_id: int | None = None
    children: list[int] = Field(default_factory=list)


class PipelineRun(BaseModel):
    """Metadata for a single pipeline execution."""

    run_id: str
    timestamp: datetime
    config_snapshot: dict
    conversation_count: int
    cluster_count: int

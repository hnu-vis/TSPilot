"""External API models."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from runtime.trace import TraceEventModel
from schemas.database_context import DatabaseContext
from schemas.output import FinalAnswer


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: str | None = None


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None
    model_id: str | None = None
    database_context: DatabaseContext | None = None
    selected_database: str | None = None
    selected_database_type: str | None = None
    time_range: dict | None = None
    constraints: dict = Field(default_factory=dict)
    history: list[Message] = Field(default_factory=list)
    stream: bool = False


class ChatResponse(BaseModel):
    conversation_id: str
    request_id: str
    status: Literal["completed", "partial", "failed"]
    response_kind: Literal["final_answer", "error"]
    used_tools: list[str] = Field(default_factory=list)
    answer: FinalAnswer | None = None
    trace: list[TraceEventModel] | None = None
    token_usage: dict | None = None
    error: str | None = None

"""Tool-call and observation models."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    tool_name: str
    tool_input: dict = Field(default_factory=dict)
    iteration: int
    reason: str | None = None


class ToolObservation(BaseModel):
    tool_name: str
    success: bool
    summary: str
    payload: dict = Field(default_factory=dict)
    error: str | None = None
    payload_truncated: bool = False
    payload_ref: str | None = None


class ReActTranscriptStep(BaseModel):
    """One structured Thought/Action/Observation memory fragment."""

    iteration: int
    question: str | None = None
    thought: str | None = None
    phase: str | None = None
    action_intention: str | None = None
    action_reason: str | None = None
    action: str
    action_input: dict = Field(default_factory=dict)
    observation: ToolObservation | None = None


class ToolError(BaseModel):
    tool_name: str
    error_code: str
    message: str
    retryable: bool = False

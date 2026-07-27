"""One parsed ReAct turn."""
from __future__ import annotations

from pydantic import BaseModel, Field


class PreviousObservationAssessment(BaseModel):
    completed_active_todo: bool = False
    reason: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    covered: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    next_action_reason: str | None = None
    can_answer: bool | None = None


class ReActTurn(BaseModel):
    thought: str
    previous_observation_assessment: PreviousObservationAssessment | None = None
    action_intention: str | None = None
    action_reason: str | None = None
    action: str
    action_input: dict


class ReActTurnParseError(BaseModel):
    error_code: str
    message: str
    raw_turn: str

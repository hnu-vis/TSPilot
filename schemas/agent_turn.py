"""One parsed ReAct turn."""
from __future__ import annotations

from pydantic import BaseModel, Field

from schemas.task_contract import TaskContract


class PreviousObservationAssessment(BaseModel):
    completed_active_todo: bool = False
    reason: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    covered: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    covered_facts: list[str] = Field(default_factory=list)
    missing_facts: list[str] = Field(default_factory=list)
    unavailable_facts: list[str] = Field(default_factory=list)
    next_fact_need: str | None = None
    completed_todos: list[int | str] = Field(default_factory=list)
    next_active_todo: int | str | None = None
    next_action_reason: str | None = None
    can_answer: bool | None = None


class ReActTurn(BaseModel):
    thought: str
    task_contract: TaskContract | None = None
    previous_observation_assessment: PreviousObservationAssessment | None = None
    action_intention: str | None = None
    action_reason: str | None = None
    action: str
    action_input: dict


class ReActTurnParseError(BaseModel):
    error_code: str
    message: str
    raw_turn: str

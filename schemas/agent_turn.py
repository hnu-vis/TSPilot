"""One parsed ReAct turn."""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from schemas.task_contract import TaskContract


class PreviousObservationAssessment(BaseModel):
    """Semantic acceptance receipt for the immediately previous Observation."""

    completed_active_todo: bool = False
    reason: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    can_answer: bool | None = None


class ReActTurn(BaseModel):
    thought: str
    task_contract: TaskContract | None = None
    previous_observation_assessment: PreviousObservationAssessment | None = None
    action: str
    action_input: dict

    @model_validator(mode="after")
    def validate_visualization_target(self):
        """Require a grounded visualization target before entering the ReAct trace."""

        if self.action == "visualization":
            source_refs = self.action_input.get("source_refs")
            if not isinstance(source_refs, list):
                raise ValueError("visualization action_input.source_refs must be a list")
            normalized_refs = [str(ref).strip() for ref in source_refs if str(ref).strip()]
            if not any(ref.startswith("insight:") for ref in normalized_refs):
                raise ValueError(
                    "visualization action_input.source_refs must include the verified target Insight"
                )
        return self


class ReActTurnParseError(BaseModel):
    error_code: str
    message: str
    raw_turn: str

"""Semantic contract shared by visualization planning and presentation."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class VisualProofObligation(BaseModel):
    """One independently inspectable evidence role required by a visual claim."""

    model_config = ConfigDict(extra="forbid")

    obligation_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    description: str = Field(min_length=1)
    required: bool


class VisualizationVerification(BaseModel):
    """Describe the claim relationship a visualization lets a user inspect."""

    model_config = ConfigDict(extra="forbid")

    target_insight_ids: list[str] = Field(default_factory=list)
    verification_question: str = Field(min_length=1)
    interpretation: str = Field(min_length=1)
    proof_obligations: list[VisualProofObligation] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_targets(self):
        self.target_insight_ids = list(dict.fromkeys(
            str(item).strip() for item in self.target_insight_ids if str(item).strip()
        ))
        obligation_ids = [item.obligation_id for item in self.proof_obligations]
        if len(obligation_ids) != len(set(obligation_ids)):
            raise ValueError("visual proof obligation ids must be unique")
        return self

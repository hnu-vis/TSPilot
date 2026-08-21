"""Semantic contract shared by visualization planning and presentation."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


VisualProofObligationId = Annotated[
    str,
    Field(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9_.-]+$"),
]


class VisualProofObligation(BaseModel):
    """One independently inspectable evidence role required by a visual claim."""

    model_config = ConfigDict(extra="forbid")

    obligation_id: VisualProofObligationId
    description: str = Field(min_length=1, max_length=800)
    required: Literal[True]


class VisualizationVerification(BaseModel):
    """Describe the claim relationship a visualization lets a user inspect."""

    model_config = ConfigDict(extra="forbid")

    target_insight_ids: list[str] = Field(default_factory=list)
    verification_question: str = Field(min_length=1, max_length=1200)
    interpretation: str = Field(min_length=1, max_length=1200)
    proof_obligations: list[VisualProofObligation] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def unique_targets(self):
        self.target_insight_ids = list(dict.fromkeys(
            str(item).strip() for item in self.target_insight_ids if str(item).strip()
        ))
        obligation_ids = [item.obligation_id for item in self.proof_obligations]
        if len(obligation_ids) != len(set(obligation_ids)):
            raise ValueError("visual proof obligation ids must be unique")
        return self

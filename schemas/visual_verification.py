"""Semantic contract shared by visualization planning and presentation."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class VisualizationVerification(BaseModel):
    """Describe the claim relationship a visualization lets a user inspect."""

    model_config = ConfigDict(extra="forbid")

    target_insight_ids: list[str] = Field(default_factory=list)
    verification_question: str = Field(min_length=1)
    interpretation: str = Field(min_length=1)

    @model_validator(mode="after")
    def unique_targets(self):
        self.target_insight_ids = list(dict.fromkeys(
            str(item).strip() for item in self.target_insight_ids if str(item).strip()
        ))
        return self

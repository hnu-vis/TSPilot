"""Task-level output contract models."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TaskContractOutput(BaseModel):
    id: str
    description: str
    output_type: str | None = None
    evidence_kind: str | None = None
    required: bool = True
    measures: list[str] = Field(default_factory=list)
    dimensions: list[str] = Field(default_factory=list)
    time_scope: dict | str | None = None
    success_criteria: str | None = None


class TaskContract(BaseModel):
    """LLM-authored contract for user-visible deliverables."""

    source: Literal["llm"] = "llm"
    goal: str
    required_outputs: list[TaskContractOutput] = Field(default_factory=list)
    constraints: dict = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    evidence_quality_notes: list[str] = Field(default_factory=list)


"""Final answer models."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from schemas.visualization import VisualizationPayload, VisualizationTemplateId


class PlannedAnswerSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    section_type: str
    heading: str | None = None
    content: str
    source_refs: list[str] = Field(default_factory=list)


class VisualIntent(BaseModel):
    """LLM-selected presentation semantics; no renderer mechanics or data arrays."""

    model_config = ConfigDict(extra="forbid")

    purpose: str
    priority: Literal["primary", "supporting"] = "primary"
    template_id: VisualizationTemplateId
    title: str
    summary: str | None = None
    source_refs: list[str] = Field(default_factory=list)
    fact_refs: list[str] = Field(default_factory=list)
    encodings: dict[str, str] = Field(default_factory=dict)


class FinalResponsePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str | None = None
    summary: str
    sections: list[PlannedAnswerSection] = Field(default_factory=list)
    visual_intents: list[VisualIntent] = Field(default_factory=list)


class AnswerSection(BaseModel):
    section_type: str
    heading: str | None = None
    content: str
    structured_payload: dict | None = None


class AnswerReference(BaseModel):
    source_type: Literal["query", "statistics", "fact", "analysis", "forecast", "anomaly", "rag", "skill"]
    source_id: str | None = None
    label: str
    evidence: dict | None = None


class AnswerClaim(BaseModel):
    """A user-visible claim with explicit grounding targets."""

    claim_id: str
    text: str
    fact_ids: list[str] = Field(default_factory=list)
    item_ids: list[str] = Field(default_factory=list)
    analysis_ids: list[str] = Field(default_factory=list)
    artifact_type: str | None = None
    artifact_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    visualization_ids: list[str] = Field(default_factory=list)


class FinalAnswer(BaseModel):
    title: str | None = None
    summary: str
    sections: list[AnswerSection] = Field(default_factory=list)
    references: list[AnswerReference] = Field(default_factory=list)
    claims: list[AnswerClaim] = Field(default_factory=list)
    visualizations: list[VisualizationPayload] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")

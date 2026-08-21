"""Final response and answer models."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from schemas.visualization import VisualizationPayload


class PlannedAnswerSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    section_type: str
    heading: str | None = None
    content: str
    source_refs: list[str] = Field(default_factory=list)


class FinalResponsePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str | None = None
    summary: str
    sections: list[PlannedAnswerSection] = Field(default_factory=list)
    visualization_ids: list[str] = Field(default_factory=list)


class AnswerSection(BaseModel):
    section_type: str
    heading: str | None = None
    content: str
    structured_payload: dict | None = None


class AnswerReference(BaseModel):
    source_type: Literal["query", "statistics", "insight", "analysis", "derived_evidence", "forecast", "anomaly", "rag", "skill"]
    source_id: str | None = None
    label: str
    evidence: dict | None = None


class AnswerClaim(BaseModel):
    claim_id: str
    text: str
    insight_ids: list[str] = Field(default_factory=list)
    item_ids: list[str] = Field(default_factory=list)
    analysis_ids: list[str] = Field(default_factory=list)
    artifact_type: str | None = None
    artifact_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    visualization_ids: list[str] = Field(default_factory=list)


class FinalAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str | None = None
    summary: str
    sections: list[AnswerSection] = Field(default_factory=list)
    references: list[AnswerReference] = Field(default_factory=list)
    claims: list[AnswerClaim] = Field(default_factory=list)
    visualizations: list[VisualizationPayload] = Field(default_factory=list)

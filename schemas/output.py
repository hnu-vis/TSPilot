"""Final answer models."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from schemas.visualization import VisualizationPayload


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


class FinalAnswer(BaseModel):
    title: str | None = None
    summary: str
    sections: list[AnswerSection] = Field(default_factory=list)
    references: list[AnswerReference] = Field(default_factory=list)
    visualizations: list[VisualizationPayload] = Field(default_factory=list)

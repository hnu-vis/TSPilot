"""Insight result models."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from schemas.visualization import VisualizationPayload


class FactCandidate(BaseModel):
    fact_id: str
    fact_type: str
    statement: str | None = None
    confidence: float | None = None
    evidence_refs: list[str] = Field(default_factory=list)


class CompletedFact(BaseModel):
    fact_id: str
    fact_type: str
    statement: str
    focus: str | None = None
    required_evidence: list[str] = Field(default_factory=list)
    evidence: dict = Field(default_factory=dict)
    confidence: float | None = None


class VerifiedFact(BaseModel):
    fact_id: str
    fact_type: str
    statement: str
    confidence: float
    evidence: dict = Field(default_factory=dict)
    verification_rule: str
    verification_status: Literal["verified"] = "verified"


class RejectedFact(BaseModel):
    fact_id: str
    fact_type: str
    statement: str | None = None
    reason: str
    evidence: dict | None = None
    verification_rule: str | None = None


class InsightResult(BaseModel):
    insight_id: str
    requested_fact_types: list[str] = Field(default_factory=list)
    supported_fact_types: list[str] = Field(default_factory=list)
    fact_candidates: list[FactCandidate] = Field(default_factory=list)
    completed_facts: list[CompletedFact] = Field(default_factory=list)
    verified_facts: list[VerifiedFact] = Field(default_factory=list)
    rejected_facts: list[RejectedFact] = Field(default_factory=list)
    summary_blocks: list[dict] = Field(default_factory=list)
    visualizations: list[VisualizationPayload] = Field(default_factory=list)
    diagnostics: dict = Field(default_factory=dict)


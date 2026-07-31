"""DataFact models for grounded data-analysis facts."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


FactStatus = Literal["verified", "unavailable", "rejected", "partial"]


class DataFactRequest(BaseModel):
    """A tool-scoped request for a fact the tool may produce."""

    name: str
    fact_type: str
    subject: str | None = None
    time_range: dict | None = None
    dimensions: dict = Field(default_factory=dict)
    requirements: dict = Field(default_factory=dict)


class FactEvidenceRef(BaseModel):
    source_type: str
    source_id: str
    label: str | None = None
    locator: dict = Field(default_factory=dict)


class DataFact(BaseModel):
    """A grounded fact produced from a tool output or marked unavailable."""

    fact_id: str
    name: str
    fact_type: str
    statement: str
    value: Any = None
    unit: str | None = None
    subject: str | None = None
    dimensions: dict = Field(default_factory=dict)
    time_range: dict | None = None
    method: str
    evidence_refs: list[FactEvidenceRef] = Field(default_factory=list)
    calculation_trace: dict = Field(default_factory=dict)
    status: FactStatus = "verified"
    confidence: float | None = None
    quality_flags: list[str] = Field(default_factory=list)
    unavailable_reason: str | None = None
    derived_from: list[str] = Field(default_factory=list)


class FactCoverage(BaseModel):
    requested: list[str] = Field(default_factory=list)
    verified: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    unavailable: list[str] = Field(default_factory=list)
    rejected: list[str] = Field(default_factory=list)
    partial: list[str] = Field(default_factory=list)


class FactSet(BaseModel):
    facts: list[DataFact] = Field(default_factory=list)
    coverage: FactCoverage = Field(default_factory=FactCoverage)


class FactEvent(BaseModel):
    iteration: int
    tool_name: str
    produced_fact_ids: list[str] = Field(default_factory=list)
    rejected_fact_ids: list[str] = Field(default_factory=list)
    unavailable_fact_ids: list[str] = Field(default_factory=list)
    coverage: FactCoverage = Field(default_factory=FactCoverage)


class FactDefinition(BaseModel):
    fact_type: str
    description: str
    required_evidence: list[str] = Field(default_factory=list)
    preferred_tool: str | None = None
    output_schema: dict = Field(default_factory=dict)
    verification_requirements: list[str] = Field(default_factory=list)
    report_guidance: str | None = None
    scope: str = "global"
    source: str = "system"
    updated_at: str | None = None


class FactRecipe(BaseModel):
    recipe_id: str
    fact_type: str
    name: str
    preferred_tool: str
    fact_request_template: dict = Field(default_factory=dict)
    expected_result_schema: dict = Field(default_factory=dict)
    verification_notes: list[str] = Field(default_factory=list)
    scope: str = "global"
    source: str = "system"
    updated_at: str | None = None


class MemoryCard(BaseModel):
    """Prompt-safe memory summary used for retrieval and UI lists."""

    id: str
    kind: str
    title: str
    description: str
    tags: list[str] = Field(default_factory=list)
    updated_at: str | None = None


class MemoryDetail(BaseModel):
    """On-demand memory payload used after a card is selected."""

    id: str
    card: MemoryCard
    fact_request: DataFactRequest | None = None
    guidance: str | None = None
    examples: list[str] = Field(default_factory=list)


class FactMemory(BaseModel):
    definitions: list[FactDefinition] = Field(default_factory=list)
    recipes: list[FactRecipe] = Field(default_factory=list)
    cards: list[MemoryCard] = Field(default_factory=list)
    details: list[MemoryDetail] = Field(default_factory=list)
    storage_path: str | None = None
    updated_at: str | None = None

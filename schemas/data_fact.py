"""DataFact models for grounded data-analysis facts."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


FactStatus = Literal["verified", "unavailable", "rejected", "partial"]


class DataFactRequest(BaseModel):
    """A semantic fact contract that one tool call may satisfy."""

    name: str
    fact_type: str
    fact_key: str | None = None
    subject: str | None = None
    time_range: dict | None = None
    dimensions: dict = Field(default_factory=dict)
    requirements: dict = Field(default_factory=dict)
    derived_from: list[str] = Field(default_factory=list)
    result_shape: str | None = None
    expected_item_count: int | None = None
    semantic_class: str | None = None
    derivation: str | None = None
    selection: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def assign_semantic_key(self):
        self.fact_key = normalize_fact_key(self.fact_key or self.name)
        self.derived_from = list(dict.fromkeys(normalize_fact_key(item) for item in self.derived_from if item))
        if self.result_shape:
            self.result_shape = str(self.result_shape).strip().lower()
        if self.expected_item_count is not None and self.expected_item_count < 0:
            raise ValueError("expected_item_count must be non-negative")
        return self


class FactEvidenceRef(BaseModel):
    source_type: str
    source_id: str
    label: str | None = None
    locator: dict = Field(default_factory=dict)


class FactItem(BaseModel):
    """A concrete observation inside a collection-valued Fact."""

    item_id: str
    value: Any = None
    label: str | None = None
    rank: int | None = None
    timestamp: str | None = None
    source_item_ids: list[str] = Field(default_factory=list)
    dimensions: dict = Field(default_factory=dict)
    evidence_refs: list[FactEvidenceRef] = Field(default_factory=list)
    locator: dict = Field(default_factory=dict)


class DataFact(BaseModel):
    """A grounded fact produced from a tool output or marked unavailable."""

    fact_id: str
    name: str
    fact_type: str
    fact_key: str | None = None
    statement: str
    value: Any = None
    value_shape: str | None = None
    items: list[FactItem] = Field(default_factory=list)
    semantic_class: str | None = None
    derivation: str | None = None
    selection: dict = Field(default_factory=dict)
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

    @model_validator(mode="after")
    def assign_semantic_key(self):
        self.fact_key = normalize_fact_key(self.fact_key or self.name)
        self.derived_from = list(dict.fromkeys(normalize_fact_key(item) for item in self.derived_from if item))
        if self.value_shape:
            self.value_shape = str(self.value_shape).strip().lower()
        return self


class FactCoverage(BaseModel):
    requested: list[str] = Field(default_factory=list)
    verified: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    unavailable: list[str] = Field(default_factory=list)
    rejected: list[str] = Field(default_factory=list)
    partial: list[str] = Field(default_factory=list)


class FactSet(BaseModel):
    requests: list[DataFactRequest] = Field(default_factory=list)
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
    description: str | None = None


FactLearningStatus = Literal["pending", "processing", "completed", "rejected", "failed"]


class FactLearningCandidate(BaseModel):
    """Value-free projection of one verified Fact usage."""

    candidate_id: str
    request_id: str
    database_id: str
    tool_name: Literal["sql_query", "code_interpreter"]
    fact_request: DataFactRequest
    fact_shape: dict = Field(default_factory=dict)
    evidence_types: list[str] = Field(default_factory=list)
    dependency_fact_keys: list[str] = Field(default_factory=list)
    calculation_semantics: dict = Field(default_factory=dict)
    created_at: str


class FactLearningJob(BaseModel):
    job_id: str
    status: FactLearningStatus = "pending"
    candidate: FactLearningCandidate
    attempt_count: int = 0
    queued_at: str
    started_at: str | None = None
    completed_at: str | None = None
    lease_expires_at: str | None = None
    failure_stage: str | None = None
    error_summary: str | None = None
    batch_id: str | None = None


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
    preferred_tool: str | None = None
    guidance: str | None = None
    examples: list[str] = Field(default_factory=list)


class FactMemory(BaseModel):
    definitions: list[FactDefinition] = Field(default_factory=list)
    recipes: list[FactRecipe] = Field(default_factory=list)
    cards: list[MemoryCard] = Field(default_factory=list)
    details: list[MemoryDetail] = Field(default_factory=list)
    storage_path: str | None = None
    updated_at: str | None = None


def normalize_fact_key(value: str) -> str:
    """Normalize a model-provided semantic key without inferring aliases."""

    text = str(value or "").strip().lower()
    normalized = "".join(character if character.isalnum() or character in {".", "_", "-"} else "_" for character in text)
    return "_".join(part for part in normalized.split("_") if part) or "fact"

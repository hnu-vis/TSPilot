"""KeyInsight models for grounded data-analysis insights."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


InsightStatus = Literal["verified", "unavailable", "rejected", "partial"]


class KeyInsightRequest(BaseModel):
    """A semantic insight contract that one tool call may satisfy."""

    name: str = Field(
        description=(
            "Concise, reusable semantic label for the insight concept. Use a canonical noun phrase, "
            "not a user instruction, full-sentence conclusion, time-scoped deliverable, or answer prose."
        )
    )
    insight_type: str
    insight_key: str | None = None
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
        self.insight_key = normalize_insight_key(self.insight_key or self.name)
        self.derived_from = list(dict.fromkeys(normalize_insight_key(item) for item in self.derived_from if item))
        if self.result_shape:
            self.result_shape = str(self.result_shape).strip().lower()
        if self.expected_item_count is not None and self.expected_item_count < 0:
            raise ValueError("expected_item_count must be non-negative")
        return self


class InsightEvidenceRef(BaseModel):
    source_type: str
    source_id: str
    label: str | None = None
    locator: dict = Field(default_factory=dict)


class InsightItem(BaseModel):
    """A concrete observation inside a collection-valued Insight."""

    item_id: str
    value: Any = None
    label: str | None = None
    rank: int | None = None
    timestamp: str | None = None
    source_item_ids: list[str] = Field(default_factory=list)
    dimensions: dict = Field(default_factory=dict)
    evidence_refs: list[InsightEvidenceRef] = Field(default_factory=list)
    locator: dict = Field(default_factory=dict)


class KeyInsight(BaseModel):
    """A grounded insight produced from a tool output or marked unavailable."""

    insight_id: str
    name: str
    insight_type: str
    insight_key: str | None = None
    statement: str
    value: Any = None
    value_shape: str | None = None
    items: list[InsightItem] = Field(default_factory=list)
    semantic_class: str | None = None
    derivation: str | None = None
    selection: dict = Field(default_factory=dict)
    unit: str | None = None
    subject: str | None = None
    dimensions: dict = Field(default_factory=dict)
    time_range: dict | None = None
    method: str
    evidence_refs: list[InsightEvidenceRef] = Field(default_factory=list)
    calculation_trace: str | dict | list = Field(default_factory=dict)
    status: InsightStatus = "verified"
    confidence: float | None = None
    quality_flags: list[str] = Field(default_factory=list)
    unavailable_reason: str | None = None
    derived_from: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def assign_semantic_key(self):
        self.insight_key = normalize_insight_key(self.insight_key or self.name)
        self.derived_from = list(dict.fromkeys(normalize_insight_key(item) for item in self.derived_from if item))
        if self.value_shape:
            self.value_shape = str(self.value_shape).strip().lower()
        return self


class InsightCoverage(BaseModel):
    requested: list[str] = Field(default_factory=list)
    verified: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    unavailable: list[str] = Field(default_factory=list)
    rejected: list[str] = Field(default_factory=list)
    partial: list[str] = Field(default_factory=list)


class InsightSet(BaseModel):
    requests: list[KeyInsightRequest] = Field(default_factory=list)
    insights: list[KeyInsight] = Field(default_factory=list)
    coverage: InsightCoverage = Field(default_factory=InsightCoverage)


class InsightEvent(BaseModel):
    iteration: int
    tool_name: str
    produced_insight_ids: list[str] = Field(default_factory=list)
    rejected_insight_ids: list[str] = Field(default_factory=list)
    unavailable_insight_ids: list[str] = Field(default_factory=list)
    coverage: InsightCoverage = Field(default_factory=InsightCoverage)


class InsightDefinition(BaseModel):
    insight_type: str
    description: str
    required_evidence: list[str] = Field(default_factory=list)
    preferred_tool: str | None = None
    output_schema: dict = Field(default_factory=dict)
    verification_requirements: list[str] = Field(default_factory=list)
    report_guidance: str | None = None
    scope: str = "global"
    source: str = "system"
    updated_at: str | None = None


class RecipeCalculationTrace(BaseModel):
    """Reusable calculation method retained from a verified runtime trace."""

    method: str = Field(min_length=1)


class InsightRecipe(BaseModel):
    recipe_id: str
    insight_type: str
    name: str = Field(
        description=(
            "Concise canonical name of the reusable Key Insight pattern. Put calculation detail in "
            "description and insight_request_template rather than expanding the name into a sentence."
        )
    )
    preferred_tool: str
    insight_request_template: dict = Field(default_factory=dict)
    expected_result_schema: dict = Field(default_factory=dict)
    verification_notes: list[str] = Field(default_factory=list)
    calculation_trace: RecipeCalculationTrace | None = None
    scope: str = "global"
    source: str = "system"
    updated_at: str | None = None
    description: str | None = None


InsightLearningStatus = Literal["pending", "processing", "completed", "rejected", "failed"]


class InsightLearningCandidate(BaseModel):
    """Value-free projection of one verified Insight usage."""

    candidate_id: str
    request_id: str
    database_id: str
    tool_name: Literal["sql_query", "code_interpreter"]
    insight_request: KeyInsightRequest
    insight_shape: dict = Field(default_factory=dict)
    evidence_types: list[str] = Field(default_factory=list)
    dependency_insight_keys: list[str] = Field(default_factory=list)
    calculation_semantics: dict = Field(default_factory=dict)
    created_at: str


class InsightLearningJob(BaseModel):
    job_id: str
    status: InsightLearningStatus = "pending"
    candidate: InsightLearningCandidate
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
    insight_request: KeyInsightRequest | None = None
    preferred_tool: str | None = None
    calculation_trace: RecipeCalculationTrace | None = None
    guidance: str | None = None
    examples: list[str] = Field(default_factory=list)


class InsightMemory(BaseModel):
    definitions: list[InsightDefinition] = Field(default_factory=list)
    recipes: list[InsightRecipe] = Field(default_factory=list)
    cards: list[MemoryCard] = Field(default_factory=list)
    details: list[MemoryDetail] = Field(default_factory=list)
    storage_path: str | None = None
    updated_at: str | None = None


def normalize_insight_key(value: str) -> str:
    """Normalize a model-provided semantic key without inferring aliases."""

    text = str(value or "").strip().lower()
    normalized = "".join(character if character.isalnum() or character in {".", "_", "-"} else "_" for character in text)
    return "_".join(part for part in normalized.split("_") if part) or "insight"

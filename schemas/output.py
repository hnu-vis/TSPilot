"""Final answer models."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from schemas.visualization import VisualizationPayload, VisualizationMark


class PlannedAnswerSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    section_type: str
    heading: str | None = None
    content: str
    source_refs: list[str] = Field(default_factory=list)


class VisualFieldEncoding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str
    data_type: Literal["time", "number", "category", "string", "boolean", "object"] | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_type_alias(cls, value):
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        if "type" in normalized and "data_type" not in normalized:
            normalized["data_type"] = normalized.pop("type")
        semantic_types = {
            "quantitative": "number",
            "temporal": "time",
            "nominal": "category",
            "ordinal": "category",
        }
        data_type = str(normalized.get("data_type") or "").strip().lower()
        if data_type in semantic_types:
            normalized["data_type"] = semantic_types[data_type]
        return normalized


class VisualFilterTransform(BaseModel):
    """A read-only row selection applied only to a visualization layer."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["filter"] = "filter"
    field: str
    operator: Literal[
        "eq", "neq", "in", "not_in", "exists", "not_exists",
        "gt", "gte", "lt", "lte", "between",
    ] = "eq"
    value: Any = None

    @model_validator(mode="after")
    def validate_value_shape(self):
        if self.operator in {"in", "not_in"} and not isinstance(self.value, list):
            raise ValueError(f"filter operator '{self.operator}' requires a list value")
        if self.operator == "between" and (not isinstance(self.value, list) or len(self.value) != 2):
            raise ValueError("filter operator 'between' requires exactly two boundary values")
        return self


class VisualLayerPlan(BaseModel):
    """One grounded semantic layer selected by the response-planning LLM."""

    model_config = ConfigDict(extra="forbid")

    role: str
    source_ref: str
    mark: VisualizationMark
    encoding: dict[str, str | VisualFieldEncoding | list[str | VisualFieldEncoding]] = Field(default_factory=dict)
    transform: list[VisualFilterTransform] = Field(default_factory=list)
    presentation: dict[str, Any] = Field(default_factory=dict)
    label: str | None = None

    @model_validator(mode="after")
    def require_renderer_mark(self):
        mark = self.mark.strip()
        if not mark:
            raise ValueError("visual layer mark must be a non-empty renderer series type")
        if mark.casefold() in {"text", "table"}:
            raise ValueError(f"'{mark}' is content, not a graphical visualization mark")
        return self


class VisualGoal(BaseModel):
    """A user-visible visual purpose and the roles required to satisfy it."""

    model_config = ConfigDict(extra="forbid")

    purpose: str
    title: str
    priority: Literal["primary", "supporting"] = "primary"
    summary: str | None = None
    required_roles: list[str] = Field(default_factory=list)
    presentation: dict[str, Any] = Field(default_factory=dict)
    layers: list[VisualLayerPlan] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_role_layer_bijection(self):
        layer_roles = {layer.role.strip().casefold() for layer in self.layers}
        missing = [
            role for role in self.required_roles
            if role.strip().casefold() not in layer_roles
        ]
        if missing:
            raise ValueError(
                f"every required visual role must have a same-role layer; missing={missing}"
            )
        return self


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
    """A user-visible claim with explicit grounding targets."""

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
    title: str | None = None
    summary: str
    sections: list[AnswerSection] = Field(default_factory=list)
    references: list[AnswerReference] = Field(default_factory=list)
    claims: list[AnswerClaim] = Field(default_factory=list)
    visualizations: list[VisualizationPayload] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")

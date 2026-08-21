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
    provenance_source_refs: list[str] = Field(default_factory=list)
    label: str | None = None

    @model_validator(mode="after")
    def require_renderer_mark(self):
        mark = self.mark.strip()
        if not mark:
            raise ValueError("visual layer mark must be a non-empty renderer series type")
        if mark.casefold() in {"text", "table"}:
            raise ValueError(f"'{mark}' is content, not a graphical visualization mark")
        return self


class VisualSemanticPlan(BaseModel):
    """One grounded semantic object with separate target and content bindings."""

    model_config = ConfigDict(extra="forbid")

    semantic_id: str = Field(min_length=1)
    semantic_type: Literal["fact", "event", "observation", "interval", "relation", "reference"]
    role: str = Field(min_length=1)
    source_ref: str = Field(min_length=1)
    target_encoding: dict[str, str] = Field(default_factory=dict)
    description_field: str | None = None
    metric_fields: list[str] = Field(default_factory=list)
    related_semantic_ids: list[str] = Field(default_factory=list)
    importance: Literal["primary", "highlight", "support"] = "support"
    line_style: Literal["solid", "dashed", "dotted"] = "solid"
    symbol: Literal["none", "circle", "diamond", "triangle", "pin"] = "none"
    presentation: dict[str, Any] = Field(default_factory=dict)
    provenance_source_refs: list[str] = Field(default_factory=list)
    label: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_semantic_shape(self):
        required = {
            "fact": set(),
            "event": {"x"},
            "observation": {"x", "y"},
            "interval": {"start", "end"},
            "relation": set(),
            "reference": {"value"},
        }[self.semantic_type]
        actual = set(self.target_encoding)
        if actual != required:
            raise ValueError(
                f"semantic '{self.semantic_id}' type '{self.semantic_type}' requires target fields "
                f"{sorted(required)}, got {sorted(actual)}"
            )
        if self.semantic_type == "relation":
            if len(self.related_semantic_ids) < 2:
                raise ValueError("relation semantic requires at least two related semantic ids")
        elif self.related_semantic_ids:
            raise ValueError("only relation semantics may contain related_semantic_ids")
        if not self.description_field and not self.metric_fields and self.semantic_type == "fact":
            raise ValueError("fact semantic requires grounded description or metric content")
        return self


class VisualGuideSectionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: str = Field(min_length=1)
    section_type: Literal["data", "semantics"]
    label: str = Field(min_length=1)
    target_ids: list[str] = Field(min_length=1)


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
    semantics: list[VisualSemanticPlan] = Field(default_factory=list)
    guides: list[VisualGuideSectionPlan] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_role_coverage(self):
        layer_roles = {
            item.role.strip().casefold()
            for item in [*self.layers, *self.semantics]
        }
        missing = [
            role for role in self.required_roles
            if role.strip().casefold() not in layer_roles
        ]
        if missing:
            raise ValueError(
                f"every required visual role must have a same-role layer; missing={missing}"
            )
        return self

    @model_validator(mode="after")
    def validate_scene_refs(self):
        semantic_ids = {item.semantic_id for item in self.semantics}
        if len(semantic_ids) != len(self.semantics):
            raise ValueError("visual semantic ids must be unique")
        for semantic in self.semantics:
            unknown = set(semantic.related_semantic_ids) - semantic_ids
            if unknown:
                raise ValueError(
                    f"relation semantic '{semantic.semantic_id}' references unknown semantics: {sorted(unknown)}"
                )
        mark_ids = {f"mark_{index}" for index, _ in enumerate(self.layers)}
        for section in self.guides:
            allowed = mark_ids if section.section_type == "data" else semantic_ids
            unknown = set(section.target_ids) - allowed
            if unknown:
                raise ValueError(
                    f"guide section '{section.section_id}' references unknown targets: {sorted(unknown)}"
                )
        return self


class VisualEncodingIR(BaseModel):
    """One closed field binding in the LLM-authored visualization IR."""

    model_config = ConfigDict(extra="forbid")

    channel: Literal["x", "y", "value", "lower", "upper", "series", "label"]
    field: str = Field(min_length=1)


class VisualSemanticIR(BaseModel):
    """Closed LLM-authored semantic intent before deterministic compilation."""

    model_config = ConfigDict(extra="forbid")

    semantic_id: str = Field(min_length=1)
    semantic_type: Literal["fact", "event", "observation", "interval", "relation", "reference"]
    role: str = Field(min_length=1)
    source_ref: str = Field(min_length=1)
    target_encodings: list[VisualEncodingIR] = Field(default_factory=list)
    description_field: str | None = None
    metric_fields: list[str] = Field(default_factory=list)
    related_semantic_ids: list[str] = Field(default_factory=list)
    importance: Literal["primary", "highlight", "support"]
    line_style: Literal["solid", "dashed", "dotted"]
    symbol: Literal["none", "circle", "diamond", "triangle", "pin"]
    label: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_semantic_contract(self):
        channels = [encoding.channel for encoding in self.target_encodings]
        if len(channels) != len(set(channels)):
            raise ValueError(f"semantic IR '{self.semantic_id}' contains duplicate target channels")
        required = {
            "fact": set(),
            "event": {"x"},
            "observation": {"x", "y"},
            "interval": {"lower", "upper"},
            "relation": set(),
            "reference": {"value"},
        }[self.semantic_type]
        if set(channels) != required:
            raise ValueError(
                f"semantic IR '{self.semantic_id}' type '{self.semantic_type}' requires "
                f"target channels {sorted(required)}, got {sorted(channels)}"
            )
        if self.semantic_type == "relation" and len(self.related_semantic_ids) < 2:
            raise ValueError("relation semantic IR requires at least two related semantic ids")
        if self.semantic_type != "relation" and self.related_semantic_ids:
            raise ValueError("only relation semantic IR may reference related semantics")
        return self


class _VisualLayerIRBase(BaseModel):
    """Fields shared by every renderer-independent layer intent."""

    model_config = ConfigDict(extra="forbid")

    role: str = Field(min_length=1)
    source_ref: str = Field(min_length=1)
    encodings: list[VisualEncodingIR]
    emphasis: Literal["normal", "subtle", "strong"]
    line_style: Literal["solid", "dashed", "dotted"]
    symbol: Literal["none", "circle", "diamond", "triangle", "pin"]
    axis: Literal["primary", "secondary"]
    label: str | None

    @model_validator(mode="after")
    def validate_layer_contract(self):
        channels = [item.channel for item in self.encodings]
        if len(channels) != len(set(channels)):
            raise ValueError(f"visual IR layer '{self.role}' contains duplicate encoding channels")
        channel_set = set(channels)
        if self.layer_type == "band":
            required = {"x", "lower", "upper"}
        elif self.layer_type == "reference_line":
            required = {"value"}
        elif self.layer_type == "annotation":
            required = {"label"}
        else:
            required = {"x", "y"}
        missing = required - channel_set
        if missing:
            raise ValueError(f"visual IR layer '{self.role}' is missing channels {sorted(missing)}")
        return self


class SeriesVisualLayerIR(_VisualLayerIRBase):
    layer_type: Literal["series"]


class EventPointsVisualLayerIR(_VisualLayerIRBase):
    layer_type: Literal["event_points"]


class BandVisualLayerIR(_VisualLayerIRBase):
    layer_type: Literal["band"]


class ComparisonVisualLayerIR(_VisualLayerIRBase):
    layer_type: Literal["comparison"]


class ReferenceLineVisualLayerIR(_VisualLayerIRBase):
    """A grounded scalar guide spanning the host plot's x domain."""

    layer_type: Literal["reference_line"]


class AnnotationVisualLayerIR(_VisualLayerIRBase):
    """Grounded content attached to a host, optionally at a real x position.

    An annotation may bind ``x`` when its source is a located event or
    observation. Omitting ``x`` keeps scalar/global content as a chart-level
    callout. In either form the annotation borrows the host coordinate system
    and never creates an axis domain of its own.
    """

    layer_type: Literal["annotation"]


class IntervalSourceVisualLayerIR(_VisualLayerIRBase):
    """An interval overlay whose one boundary pair comes from another view."""

    layer_type: Literal["interval_overlay"]
    interval_source_ref: str = Field(min_length=1)
    interval_start_field: str = Field(min_length=1)
    interval_end_field: str = Field(min_length=1)


class IntervalLiteralVisualLayerIR(_VisualLayerIRBase):
    """An interval overlay whose boundary pair is explicitly grounded in the IR."""

    layer_type: Literal["interval_overlay"]
    interval_start_value: str | int | float
    interval_end_value: str | int | float


# A plain union emits provider-compatible JSON Schema ``anyOf`` while each
# branch makes irrelevant interval properties impossible to emit.
VisualLayerIR = (
    SeriesVisualLayerIR
    | EventPointsVisualLayerIR
    | BandVisualLayerIR
    | ComparisonVisualLayerIR
    | ReferenceLineVisualLayerIR
    | AnnotationVisualLayerIR
    | IntervalSourceVisualLayerIR
    | IntervalLiteralVisualLayerIR
)


class VisualGoalIR(BaseModel):
    """Closed chart goal authored by the LLM before deterministic compilation."""

    model_config = ConfigDict(extra="forbid")

    purpose: str = Field(min_length=1)
    title: str = Field(min_length=1)
    priority: Literal["primary", "supporting"]
    summary: str | None
    required_roles: list[str]
    show_legend: bool
    tooltip: Literal["axis", "item", "none"]
    enable_zoom: bool
    viewport_start: str | int | float | None
    viewport_end: str | int | float | None
    y_scale: Literal["linear", "log"]
    layers: list[VisualLayerIR] = Field(min_length=1)
    semantics: list[VisualSemanticIR] = Field(default_factory=list)
    guides: list[VisualGuideSectionPlan] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_role_coverage(self):
        layer_roles = {
            item.role.strip().casefold()
            for item in [*self.layers, *self.semantics]
        }
        missing = [
            role for role in self.required_roles
            if role.strip().casefold() not in layer_roles
        ]
        if missing:
            raise ValueError(f"every required visual role requires a same-role IR layer; missing={missing}")
        if (self.viewport_start is None) != (self.viewport_end is None):
            raise ValueError("visual viewport requires both start and end")
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

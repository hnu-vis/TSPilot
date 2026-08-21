"""Versioned, renderer-independent visualization scene contracts."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from schemas.visual_verification import VisualizationVerification


VisualizationMark = str
SemanticType = Literal["fact", "event", "observation", "interval", "relation", "reference"]


class VisualizationBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    binding_id: str
    source_type: str
    insight_id: str | None = None
    item_id: str | None = None
    related_item_ids: list[str] = Field(default_factory=list)
    evidence_id: str | None = None
    source_ref: str | None = None
    locator: dict = Field(default_factory=dict)


class VisualizationField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    data_type: Literal["time", "number", "category", "string", "boolean"]
    semantic_role: str = Field(min_length=1)
    unit: str | None = None


class VisualizationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: str = Field(min_length=1)
    values: dict[str, Any]
    binding_id: str | None = None


class VisualizationDataView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    view_id: str = Field(min_length=1)
    source_ref: str = Field(min_length=1)
    data_ref: str | None = None
    row_count: int | None = None
    time_range: dict | None = None
    fields: list[VisualizationField] = Field(default_factory=list)
    records: list[VisualizationRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_records(self):
        names = {field.name for field in self.fields}
        unknown = {
            key
            for record in self.records
            for key in record.values
            if key not in names
        }
        if unknown:
            raise ValueError(f"visual data view '{self.view_id}' contains unknown fields: {sorted(unknown)}")
        record_ids = [record.record_id for record in self.records]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError(f"visual data view '{self.view_id}' record ids must be unique")
        return self


class VisualizationScale(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scale_id: str = Field(min_length=1)
    channel: Literal["x", "y"]
    data_type: Literal["time", "number", "category", "string"]
    semantic_role: str = Field(min_length=1)
    unit: str | None = None
    scale_type: Literal["time", "category", "linear", "log"]


class VisualizationCoordinateSpace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    space_id: str = Field(min_length=1)
    coordinate: Literal["cartesian"] = "cartesian"
    x_scale_id: str = Field(min_length=1)
    y_scale_ids: list[str] = Field(min_length=1)


class VisualizationChannelBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fields: list[str] = Field(min_length=1)
    scale_id: str | None = None


class VisualizationDataMark(BaseModel):
    """An axis-bearing data mark; semantic guides never become fake marks."""

    model_config = ConfigDict(extra="forbid")

    mark_id: str = Field(min_length=1)
    mark: VisualizationMark
    role: str = Field(min_length=1)
    source_ref: str = Field(min_length=1)
    data_view_id: str = Field(min_length=1)
    space_id: str = Field(min_length=1)
    encoding: dict[str, VisualizationChannelBinding] = Field(default_factory=dict)
    transform: list[dict] = Field(default_factory=list)
    presentation: dict[str, Any] = Field(default_factory=dict)
    label: str | None = None

    @model_validator(mode="after")
    def require_data_mark(self):
        mark = self.mark.strip().casefold()
        if not mark or mark in {"text", "table", "annotation", "rule", "rect"}:
            raise ValueError("visualization data mark requires an axis-bearing graphical mark")
        return self


class VisualizationMetric(BaseModel):
    """Raw typed content; locale formatting belongs to the renderer."""

    model_config = ConfigDict(extra="forbid")

    label: str | None = None
    value: Any
    data_type: Literal["time", "number", "category", "string", "boolean"]
    unit: str | None = None


class VisualizationSemanticContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    description: str | None = None
    metrics: list[VisualizationMetric] = Field(default_factory=list)


class ChartSemanticTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_type: Literal["chart"] = "chart"


class XSemanticTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_type: Literal["x"] = "x"
    space_id: str
    scale_id: str
    record_id: str
    x: Any


class XYSemanticTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_type: Literal["xy"] = "xy"
    space_id: str
    x_scale_id: str
    y_scale_id: str
    record_id: str
    x: Any
    y: float


class IntervalSemanticTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_type: Literal["interval"] = "interval"
    space_id: str
    scale_id: str
    axis: Literal["x", "y"]
    record_id: str
    start: Any
    end: Any


class RelationSemanticTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_type: Literal["relation"] = "relation"
    semantic_ids: list[str] = Field(min_length=2)


class ReferenceSemanticTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_type: Literal["reference"] = "reference"
    space_id: str
    scale_id: str
    axis: Literal["x", "y"]
    record_id: str
    value: Any


SemanticTarget = (
    ChartSemanticTarget
    | XSemanticTarget
    | XYSemanticTarget
    | IntervalSemanticTarget
    | RelationSemanticTarget
    | ReferenceSemanticTarget
)


class VisualizationSemantic(BaseModel):
    """One grounded semantic object with independent target and content."""

    model_config = ConfigDict(extra="forbid")

    semantic_id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    semantic_type: SemanticType
    role: str = Field(min_length=1)
    source_ref: str = Field(min_length=1)
    target: SemanticTarget
    content: VisualizationSemanticContent
    importance: Literal["primary", "highlight", "support"]
    line_style: Literal["solid", "dashed", "dotted"] = "solid"
    symbol: Literal["none", "circle", "diamond", "triangle", "pin"] = "none"
    presentation: dict[str, Any] = Field(default_factory=dict)
    binding_id: str | None = None

    @model_validator(mode="after")
    def validate_target_type(self):
        expected = {
            "fact": "chart",
            "event": "x",
            "observation": "xy",
            "interval": "interval",
            "relation": "relation",
            "reference": "reference",
        }[self.semantic_type]
        if self.target.target_type != expected:
            raise ValueError(
                f"semantic type '{self.semantic_type}' requires target_type '{expected}', "
                f"got '{self.target.target_type}'"
            )
        return self


class VisualizationGuideEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    target_type: Literal["mark", "semantic"]
    target_id: str = Field(min_length=1)
    interaction: Literal["toggle", "select", "none"]
    swatch: Literal[
        "line", "point", "bar", "area", "band", "mark", "event",
        "observation", "interval", "relation", "reference", "fact",
    ]
    binding_id: str | None = None


class VisualizationGuideSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: str = Field(min_length=1)
    section_type: Literal["data", "semantics"]
    label: str = Field(min_length=1)
    entries: list[VisualizationGuideEntry] = Field(min_length=1)


class VisualizationGuide(BaseModel):
    model_config = ConfigDict(extra="forbid")

    guide_id: str = Field(min_length=1)
    guide_type: Literal["legend"] = "legend"
    position: Literal["top", "bottom"] = "top"
    sections: list[VisualizationGuideSection] = Field(min_length=1)


class VisualizationLayoutCell(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cell_id: str = Field(min_length=1)
    space_ids: list[str] = Field(min_length=1)


class VisualizationLayout(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["overlay", "facets"] = "overlay"
    cells: list[VisualizationLayoutCell] = Field(min_length=1)


class VisualizationAccessibility(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str
    table_columns: list[str] = Field(default_factory=list)
    table_rows: list[dict] = Field(default_factory=list)


class VisualizationPayload(BaseModel):
    """Public V4 payload: a grounded semantic scene, never renderer code."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["4"] = "4"
    visualization_id: str
    data_ref: str | None = None
    purpose: str
    priority: Literal["primary", "supporting"] = "primary"
    title: str
    summary: str | None = None
    verification: VisualizationVerification | None = None
    source_refs: list[str] = Field(default_factory=list)
    required_roles: list[str] = Field(default_factory=list)
    data_views: list[VisualizationDataView] = Field(default_factory=list)
    scales: list[VisualizationScale] = Field(default_factory=list)
    coordinate_spaces: list[VisualizationCoordinateSpace] = Field(default_factory=list)
    marks: list[VisualizationDataMark] = Field(default_factory=list)
    semantics: list[VisualizationSemantic] = Field(default_factory=list)
    guides: list[VisualizationGuide] = Field(default_factory=list)
    layout: VisualizationLayout
    bindings: list[VisualizationBinding] = Field(default_factory=list)
    presentation: dict[str, Any] = Field(default_factory=dict)
    accessibility: VisualizationAccessibility

    @model_validator(mode="after")
    def validate_scene(self):
        if not self.marks and not self.semantics and not self.data_ref:
            raise ValueError("visualization payload must contain a mark or semantic object")
        self._validate_bindings_and_views()
        scale_ids, space_ids = self._validate_spaces()
        mark_ids = self._validate_marks(scale_ids, space_ids)
        semantic_ids, group_ids = self._validate_semantics(space_ids)
        self._validate_guides(mark_ids, group_ids)
        if len(semantic_ids) != len(self.semantics):
            raise ValueError("visualization semantic ids must be unique")
        return self

    def _validate_bindings_and_views(self) -> None:
        binding_ids = {binding.binding_id for binding in self.bindings}
        referenced = {
            record.binding_id
            for view in self.data_views
            for record in view.records
            if record.binding_id
        } | {semantic.binding_id for semantic in self.semantics if semantic.binding_id}
        unknown = referenced - binding_ids
        if unknown:
            raise ValueError(f"visualization scene references unknown bindings: {sorted(unknown)}")
        view_ids = [view.view_id for view in self.data_views]
        if len(view_ids) != len(set(view_ids)):
            raise ValueError("visualization data-view ids must be unique")
        missing = {mark.data_view_id for mark in self.marks} - set(view_ids)
        if missing:
            raise ValueError(f"visualization marks reference unknown data views: {sorted(missing)}")

    def _validate_spaces(self) -> tuple[set[str], set[str]]:
        scale_ids = {scale.scale_id for scale in self.scales}
        space_ids = {space.space_id for space in self.coordinate_spaces}
        if len(scale_ids) != len(self.scales) or len(space_ids) != len(self.coordinate_spaces):
            raise ValueError("visualization scale and coordinate-space ids must be unique")
        for space in self.coordinate_spaces:
            missing = {space.x_scale_id, *space.y_scale_ids} - scale_ids
            if missing:
                raise ValueError(f"coordinate space '{space.space_id}' references unknown scales: {sorted(missing)}")
        return scale_ids, space_ids

    def _validate_marks(self, scale_ids: set[str], space_ids: set[str]) -> set[str]:
        mark_ids = {mark.mark_id for mark in self.marks}
        if len(mark_ids) != len(self.marks):
            raise ValueError("visualization mark ids must be unique")
        views = {view.view_id: view for view in self.data_views}
        for mark in self.marks:
            if mark.space_id not in space_ids:
                raise ValueError(f"mark '{mark.mark_id}' references unknown space '{mark.space_id}'")
            field_names = {field.name for field in views[mark.data_view_id].fields}
            for channel, binding in mark.encoding.items():
                unknown = set(binding.fields) - field_names
                if unknown:
                    raise ValueError(f"mark '{mark.mark_id}' channel '{channel}' uses unknown fields: {sorted(unknown)}")
                if binding.scale_id is not None and binding.scale_id not in scale_ids:
                    raise ValueError(f"mark '{mark.mark_id}' references unknown scale '{binding.scale_id}'")
        return mark_ids

    def _validate_semantics(self, space_ids: set[str]) -> tuple[set[str], set[str]]:
        semantic_ids = {semantic.semantic_id for semantic in self.semantics}
        group_ids = {semantic.group_id for semantic in self.semantics}
        for semantic in self.semantics:
            target = semantic.target
            if target.target_type == "relation":
                unknown = set(target.semantic_ids) - semantic_ids
                if unknown:
                    raise ValueError(f"relation '{semantic.semantic_id}' references unknown members: {sorted(unknown)}")
            elif target.target_type != "chart" and target.space_id not in space_ids:
                raise ValueError(f"semantic '{semantic.semantic_id}' references unknown space '{target.space_id}'")
        return semantic_ids, group_ids

    def _validate_guides(self, mark_ids: set[str], group_ids: set[str]) -> None:
        for guide in self.guides:
            for section in guide.sections:
                for entry in section.entries:
                    known = mark_ids if entry.target_type == "mark" else group_ids
                    if entry.target_id not in known:
                        raise ValueError(
                            f"guide entry '{entry.entry_id}' references unknown {entry.target_type} '{entry.target_id}'"
                        )

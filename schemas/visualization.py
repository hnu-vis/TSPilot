"""LineChart-first, renderer-independent visualization contract (V4)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from schemas.visual_verification import VisualizationVerification

VisualizationMark = str
FieldType = Literal["time", "number", "category", "string", "boolean"]
Importance = Literal["primary", "highlight", "support"]


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
    data_type: FieldType
    semantic_role: str = Field(min_length=1)
    measure: str | None = None
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
    fields: list[VisualizationField] = Field(min_length=1)
    records: list[VisualizationRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_records(self):
        field_names = {field.name for field in self.fields}
        unknown = {name for record in self.records for name in record.values if name not in field_names}
        if unknown:
            raise ValueError(f"data view '{self.view_id}' contains unknown fields: {sorted(unknown)}")
        record_ids = [record.record_id for record in self.records]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError(f"data view '{self.view_id}' record ids must be unique")
        return self


class LineChartXAxis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    axis_id: str = "x"
    data_type: Literal["time", "category"]
    label: str | None = None


class LineChartYAxis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    axis_id: str = Field(min_length=1)
    label: str | None = None
    measure: str = Field(min_length=1)
    unit: str | None = None
    scale: Literal["linear", "log"] = "linear"


class _ComponentBase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    component_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    importance: Importance
    source_ref: str = Field(min_length=1)
    view_id: str = Field(min_length=1)
    label: str | None = None
    binding_ids: list[str] = Field(default_factory=list)
    presentation: dict[str, Any] = Field(default_factory=dict)


class LineChartLine(_ComponentBase):
    x_field: str = Field(min_length=1)
    y_field: str = Field(min_length=1)
    y_axis_id: str = Field(min_length=1)
    line_style: Literal["solid", "dashed", "dotted"] = "solid"
    symbol: Literal["none", "circle", "diamond", "triangle"] = "none"


class LineChartPoint(_ComponentBase):
    x_field: str = Field(min_length=1)
    y_field: str = Field(min_length=1)
    y_axis_id: str = Field(min_length=1)
    symbol: Literal["circle", "diamond", "triangle", "pin"] = "circle"
    size: Literal["small", "medium", "large"] = "medium"


class LineChartBand(_ComponentBase):
    x_field: str = Field(min_length=1)
    lower_field: str = Field(min_length=1)
    upper_field: str = Field(min_length=1)
    y_axis_id: str = Field(min_length=1)


class LineChartInterval(_ComponentBase):
    start_field: str = Field(min_length=1)
    end_field: str = Field(min_length=1)


class LineChartReferenceLine(_ComponentBase):
    value_field: str = Field(min_length=1)
    y_axis_id: str = Field(min_length=1)


class ChartAnnotationTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_type: Literal["chart"] = "chart"


class XAnnotationTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_type: Literal["x"] = "x"
    x_field: str = Field(min_length=1)


class XYAnnotationTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_type: Literal["xy"] = "xy"
    x_field: str = Field(min_length=1)
    y_field: str = Field(min_length=1)
    y_axis_id: str = Field(min_length=1)


class IntervalAnnotationTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_type: Literal["interval"] = "interval"
    start_field: str = Field(min_length=1)
    end_field: str = Field(min_length=1)


AnnotationTarget = ChartAnnotationTarget | XAnnotationTarget | XYAnnotationTarget | IntervalAnnotationTarget


class LineChartAnnotation(_ComponentBase):
    content_field: str = Field(min_length=1)
    target: AnnotationTarget


class LineChartLegend(BaseModel):
    model_config = ConfigDict(extra="forbid")
    visible: bool = True
    toggle_components: bool = True
    position: Literal["top", "bottom"] = "top"


class LineChartTooltip(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["axis", "item", "none"] = "axis"
    show_source: bool = True


class LineChartZoom(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    start: Any = None
    end: Any = None

    @model_validator(mode="after")
    def validate_range(self):
        if (self.start is None) != (self.end is None):
            raise ValueError("zoom requires both start and end")
        return self


class VisualizationAccessibility(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str
    table_columns: list[str] = Field(default_factory=list)
    table_rows: list[dict] = Field(default_factory=list)


class VisualizationPayload(BaseModel):
    """One grounded LineChart and its standard interactions."""
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["4"] = "4"
    chart_type: Literal["line"] = "line"
    visualization_id: str
    data_ref: str | None = None
    purpose: str
    priority: Literal["primary", "supporting"] = "primary"
    title: str
    summary: str | None = None
    verification: VisualizationVerification | None = None
    source_refs: list[str] = Field(default_factory=list)
    required_roles: list[str] = Field(default_factory=list)
    data_views: list[VisualizationDataView] = Field(min_length=1)
    x_axis: LineChartXAxis
    y_axes: list[LineChartYAxis] = Field(min_length=1)
    lines: list[LineChartLine] = Field(min_length=1)
    points: list[LineChartPoint] = Field(default_factory=list)
    bands: list[LineChartBand] = Field(default_factory=list)
    intervals: list[LineChartInterval] = Field(default_factory=list)
    reference_lines: list[LineChartReferenceLine] = Field(default_factory=list)
    annotations: list[LineChartAnnotation] = Field(default_factory=list)
    legend: LineChartLegend = Field(default_factory=LineChartLegend)
    tooltip: LineChartTooltip = Field(default_factory=LineChartTooltip)
    zoom: LineChartZoom = Field(default_factory=LineChartZoom)
    bindings: list[VisualizationBinding] = Field(default_factory=list)
    accessibility: VisualizationAccessibility

    @model_validator(mode="after")
    def validate_references(self):
        view_ids = {view.view_id for view in self.data_views}
        axis_ids = {axis.axis_id for axis in self.y_axes}
        binding_ids = {binding.binding_id for binding in self.bindings}
        components = [*self.lines, *self.points, *self.bands, *self.intervals, *self.reference_lines, *self.annotations]
        component_ids = [item.component_id for item in components]
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("line-chart component ids must be unique")
        for component in components:
            if component.view_id not in view_ids:
                raise ValueError(f"component '{component.component_id}' references unknown view")
            unknown_bindings = set(component.binding_ids) - binding_ids
            if unknown_bindings:
                raise ValueError(f"component '{component.component_id}' references unknown bindings: {sorted(unknown_bindings)}")
        for component in [*self.lines, *self.points, *self.bands, *self.reference_lines]:
            if component.y_axis_id not in axis_ids:
                raise ValueError(f"component '{component.component_id}' references unknown y axis")
        return self

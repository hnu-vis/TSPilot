"""Versioned, renderer-independent visualization contracts."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


VisualizationTemplateId = Literal[
    "metric.single",
    "table.detail",
    "ranking.topk",
    "timeseries.trend",
    "timeseries.highlight",
    "interval.highlight",
    "timeseries.forecast",
    "timeseries.anomaly",
    "category.comparison",
    "timeseries.comparison",
    "distribution.histogram",
    "distribution.boxplot",
    "relationship.scatter",
]


class VisualizationBinding(BaseModel):
    """Bind one semantic visual mark to its grounded source."""

    binding_id: str
    source_type: str
    fact_id: str | None = None
    item_id: str | None = None
    related_item_ids: list[str] = Field(default_factory=list)
    evidence_id: str | None = None
    source_ref: str | None = None
    locator: dict = Field(default_factory=dict)


class VisualizationDimension(BaseModel):
    name: str
    data_type: Literal["time", "number", "category", "string"]
    role: Literal["x", "y", "series", "label", "lower", "upper", "value"]
    unit: str | None = None


class VisualizationPoint(BaseModel):
    x: Any = None
    y: float | None = None
    lower: float | None = None
    upper: float | None = None
    label: str | None = None
    binding_id: str | None = None
    metadata: dict = Field(default_factory=dict)


class VisualizationSeries(BaseModel):
    series_id: str
    name: str
    role: Literal[
        "historical",
        "forecast",
        "comparison",
        "ranking",
        "distribution",
        "relationship",
    ]
    unit: str | None = None
    points: list[VisualizationPoint] = Field(default_factory=list)


class VisualizationDataset(BaseModel):
    dimensions: list[VisualizationDimension] = Field(default_factory=list)
    series: list[VisualizationSeries] = Field(default_factory=list)
    rows: list[dict] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    metric: dict | None = None


class VisualizationLayer(BaseModel):
    kind: Literal["line", "bar", "point", "rule", "area", "band", "boxplot", "scatter"]
    role: Literal["context", "fact", "forecast", "anomaly", "comparison", "confidence"]
    series_id: str | None = None
    points: list[VisualizationPoint] = Field(default_factory=list)
    label: str | None = None


class VisualizationAccessibility(BaseModel):
    description: str
    table_columns: list[str] = Field(default_factory=list)
    table_rows: list[dict] = Field(default_factory=list)


class VisualizationPayload(BaseModel):
    """Public V2 payload; it contains semantics and data, never renderer code."""

    schema_version: Literal["2"] = "2"
    visualization_id: str
    template_id: VisualizationTemplateId
    purpose: str
    priority: Literal["primary", "supporting"] = "primary"
    title: str
    summary: str | None = None
    source_refs: list[str] = Field(default_factory=list)
    fact_refs: list[str] = Field(default_factory=list)
    dataset: VisualizationDataset = Field(default_factory=VisualizationDataset)
    layers: list[VisualizationLayer] = Field(default_factory=list)
    bindings: list[VisualizationBinding] = Field(default_factory=list)
    layout: Literal["overlay", "facets"] = "overlay"
    accessibility: VisualizationAccessibility

    @model_validator(mode="after")
    def validate_renderable_content(self):
        has_content = bool(
            self.dataset.series
            or self.dataset.rows
            or self.dataset.metric
            or any(layer.points for layer in self.layers)
        )
        if not has_content:
            raise ValueError("visualization payload must contain renderable data")
        binding_ids = {binding.binding_id for binding in self.bindings}
        referenced_ids = {
            point.binding_id
            for series in self.dataset.series
            for point in series.points
            if point.binding_id
        }
        referenced_ids.update(
            point.binding_id
            for layer in self.layers
            for point in layer.points
            if point.binding_id
        )
        unknown = referenced_ids - binding_ids
        if unknown:
            raise ValueError(f"visualization points reference unknown bindings: {sorted(unknown)}")
        return self

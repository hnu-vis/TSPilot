"""Versioned, renderer-independent visualization contracts."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


# The renderer owns the available series vocabulary.  Keeping this as a string
# prevents the API contract from becoming the bottleneck every time the
# renderer gains a new visual form (candlestick, heatmap, sankey, ...).
VisualizationMark = str


class VisualizationBinding(BaseModel):
    """Bind one semantic visual mark to its grounded source."""

    binding_id: str
    source_type: str
    insight_id: str | None = None
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
    role: str
    unit: str | None = None
    points: list[VisualizationPoint] = Field(default_factory=list)


class VisualizationDataset(BaseModel):
    dataset_id: str
    source_ref: str
    data_ref: str | None = None
    row_count: int | None = None
    time_range: dict | None = None
    dimensions: list[VisualizationDimension] = Field(default_factory=list)
    series: list[VisualizationSeries] = Field(default_factory=list)


class VisualizationLayer(BaseModel):
    layer_id: str
    mark: VisualizationMark
    role: str
    source_ref: str
    encoding: dict[str, str | list[str]] = Field(default_factory=dict)
    transform: list[dict] = Field(default_factory=list)
    presentation: dict[str, Any] = Field(default_factory=dict)
    dataset_id: str
    series_id: str | None = None
    points: list[VisualizationPoint] = Field(default_factory=list)
    label: str | None = None

    @model_validator(mode="after")
    def require_graphical_mark(self):
        mark = self.mark.strip()
        if not mark or mark.casefold() in {"text", "table"}:
            raise ValueError("visualization layer requires a graphical renderer mark")
        return self


class VisualizationAccessibility(BaseModel):
    description: str
    table_columns: list[str] = Field(default_factory=list)
    table_rows: list[dict] = Field(default_factory=list)


class VisualizationPayload(BaseModel):
    """Public V3 payload; it contains grounded layers, never renderer code."""

    schema_version: Literal["3"] = "3"
    visualization_id: str
    data_ref: str | None = None
    purpose: str
    priority: Literal["primary", "supporting"] = "primary"
    title: str
    summary: str | None = None
    source_refs: list[str] = Field(default_factory=list)
    required_roles: list[str] = Field(default_factory=list)
    datasets: list[VisualizationDataset] = Field(default_factory=list)
    layers: list[VisualizationLayer] = Field(default_factory=list)
    bindings: list[VisualizationBinding] = Field(default_factory=list)
    layout: Literal["overlay", "facets"] = "overlay"
    presentation: dict[str, Any] = Field(default_factory=dict)
    accessibility: VisualizationAccessibility

    @model_validator(mode="after")
    def validate_renderable_content(self):
        has_content = bool(
            any(dataset.series for dataset in self.datasets)
            or any(layer.points for layer in self.layers)
        )
        if not has_content and not self.data_ref:
            raise ValueError("visualization payload must contain renderable data")
        binding_ids = {binding.binding_id for binding in self.bindings}
        referenced_ids = {
            point.binding_id
            for dataset in self.datasets
            for series in dataset.series
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
        dataset_ids = {dataset.dataset_id for dataset in self.datasets}
        missing_datasets = {layer.dataset_id for layer in self.layers} - dataset_ids
        if missing_datasets:
            raise ValueError(f"visualization layers reference unknown datasets: {sorted(missing_datasets)}")
        return self

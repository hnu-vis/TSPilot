"""Public native-ECharts visualization artifact contract (V5)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from schemas.visual_verification import VisualizationVerification


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


class VisualizationAccessibility(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str
    table_columns: list[str] = Field(default_factory=list)
    table_rows: list[dict[str, Any]] = Field(default_factory=list)


class VisualizationPayload(BaseModel):
    """A complete, grounded ECharts option plus evidence bindings."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["5"] = "5"
    chart_type: Literal["echarts"] = "echarts"
    visualization_id: str
    data_ref: str | None = None
    purpose: str
    priority: Literal["primary", "supporting"] = "primary"
    title: str
    summary: str | None = None
    warnings: list[str] = Field(default_factory=list)
    verification: VisualizationVerification | None = None
    option: dict[str, Any]
    source_refs: list[str] = Field(default_factory=list)
    bindings: list[VisualizationBinding] = Field(default_factory=list)
    accessibility: VisualizationAccessibility

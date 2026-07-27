"""Visualization payload model."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class VisualizationPayload(BaseModel):
    """Structured visualization payload."""

    visualization_id: str
    visualization_type: Literal["chart", "table", "metric_card", "annotation"]
    visualization_kind: str
    renderer: str
    title: str
    summary: str | None = None
    chart: dict | None = None
    annotations: list[dict] = Field(default_factory=list)
    binding_fact_ids: list[str] = Field(default_factory=list)
    binding_evidence_ids: list[str] = Field(default_factory=list)
    requested_capabilities: list[str] = Field(default_factory=list)
    requested_fact_types: list[str] = Field(default_factory=list)
    subject: dict = Field(default_factory=dict)
    presentation: dict = Field(default_factory=dict)
    row_count: int | None = None
    columns: list[str] = Field(default_factory=list)
    rows: list[dict] = Field(default_factory=list)
    display_rows: list[dict] = Field(default_factory=list)
    time_column: str | None = None
    primary_measure: str | None = None
    legend: list[dict] = Field(default_factory=list)
    display_priority: int = 0
    render_hints: dict = Field(default_factory=dict)

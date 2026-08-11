"""Time-series result models."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

class TimeSeriesPoint(BaseModel):
    timestamp: str
    value: float


class TimeSeriesSeries(BaseModel):
    series_name: str | None = None
    time_field: str
    value_field: str
    points: list[TimeSeriesPoint] = Field(default_factory=list)
    labels: dict = Field(default_factory=dict)


class ForecastPlan(BaseModel):
    mode: Literal["direct", "rolling", "requires_rolling"]
    horizon_source: Literal["explicit_steps", "duration_from_user", "inferred_short_term_default"]
    requested_steps: int
    resolved_steps: int
    sampling_interval_seconds: int | None = None
    forecast_duration_seconds: int | None = None
    forecast_start: str | None = None
    forecast_end: str | None = None
    max_direct_steps: int = 48
    recommended_chunk_steps: int | None = None
    reason: str | None = None


class ForecastResult(BaseModel):
    forecast_id: str
    model_name: str
    horizon: int
    status: Literal["succeeded", "requires_rolling"] = "succeeded"
    forecast_plan: ForecastPlan | None = None
    forecast_points: list[TimeSeriesPoint] = Field(default_factory=list)
    confidence_interval: list[dict] = Field(default_factory=list)
    diagnostics: dict = Field(default_factory=dict)


class AnomalyResult(BaseModel):
    anomaly_id: str
    detector_name: str
    anomaly_points: list[dict] = Field(default_factory=list)
    anomaly_spans: list[dict] = Field(default_factory=list)
    scores: list[dict] = Field(default_factory=list)
    diagnostics: dict = Field(default_factory=dict)

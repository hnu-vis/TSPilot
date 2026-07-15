"""Time-series result models."""
from __future__ import annotations

from pydantic import BaseModel, Field

from schemas.visualization import VisualizationPayload


class TimeSeriesPoint(BaseModel):
    timestamp: str
    value: float


class TimeSeriesSeries(BaseModel):
    series_name: str | None = None
    time_field: str
    value_field: str
    points: list[TimeSeriesPoint] = Field(default_factory=list)
    labels: dict = Field(default_factory=dict)


class ForecastResult(BaseModel):
    forecast_id: str
    model_name: str
    horizon: int
    forecast_points: list[TimeSeriesPoint] = Field(default_factory=list)
    confidence_interval: list[dict] = Field(default_factory=list)
    diagnostics: dict = Field(default_factory=dict)
    visualizations: list[VisualizationPayload] = Field(default_factory=list)


class AnomalyResult(BaseModel):
    anomaly_id: str
    detector_name: str
    anomaly_points: list[dict] = Field(default_factory=list)
    anomaly_spans: list[dict] = Field(default_factory=list)
    scores: list[dict] = Field(default_factory=list)
    diagnostics: dict = Field(default_factory=dict)
    visualizations: list[VisualizationPayload] = Field(default_factory=list)


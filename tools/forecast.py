"""Forecast tool placeholder."""
from __future__ import annotations

import re

from pydantic import BaseModel, Field
from pydantic import field_validator

from core.timeseries.forecast_adapter import linear_forecast
from core.timeseries.normalization import normalize_timeseries_evidence
from schemas.database import DatabaseEvidence
from schemas.timeseries import ForecastResult
from schemas.visualization import VisualizationPayload
from tools.base import BaseTool


class ForecastInput(BaseModel):
    database_evidence: DatabaseEvidence | dict | str | None = None
    horizon: int | None = None
    series_name: str | None = None
    constraints: dict | None = Field(default_factory=dict)

    @field_validator("horizon", mode="before")
    @classmethod
    def normalize_horizon(cls, value):
        if isinstance(value, dict):
            for key in ("steps", "horizon", "points", "count"):
                if key in value:
                    return cls.normalize_horizon(value[key])
        if isinstance(value, str):
            match = re.search(r"\d+", value)
            if match:
                return int(match.group(0))
        return value


class ForecastTool(BaseTool):
    async def execute(self, validated_input: ForecastInput, **kwargs) -> dict:
        request_state = kwargs.get("request_state")
        database_evidence = validated_input.database_evidence
        if request_state is not None:
            database_evidence = _resolve_database_evidence(database_evidence, request_state)
        if database_evidence is None:
            raise ValueError("Forecast requires database_evidence or a latest_database_evidence in request state.")
        preferred_series = validated_input.series_name or validated_input.constraints.get("series_name")
        series = normalize_timeseries_evidence(
            database_evidence,
            series_name=preferred_series,
            value_field=preferred_series,
        )
        horizon = validated_input.horizon or int(validated_input.constraints.get("horizon", 12))
        forecast_points = linear_forecast(series, horizon)
        visualization = VisualizationPayload(
            visualization_id=f"viz_forecast_{database_evidence.evidence_id}",
            visualization_type="chart",
            visualization_kind="line",
            renderer="linechart",
            title=f"{series.value_field} forecast",
            summary=f"{series.value_field} 的未来 {horizon} 个点预测。",
            chart={
                "x_axis_data": [point.timestamp for point in [*series.points, *forecast_points]],
                "series_data": [
                    {
                        "name": "historical",
                        "data": [point.value for point in series.points],
                    },
                    {
                        "name": "forecast",
                        "data": [point.value for point in forecast_points],
                    },
                ],
            },
            binding_evidence_ids=[database_evidence.evidence_id],
            requested_fact_types=["forecast"],
            time_column=series.time_field,
            primary_measure=series.value_field,
            display_priority=2,
        )
        return ForecastResult(
            forecast_id=f"forecast_{database_evidence.evidence_id}",
            model_name="linear_regression",
            horizon=horizon,
            forecast_points=forecast_points,
            confidence_interval=[],
            diagnostics={"series_name": series.series_name},
            visualizations=[visualization],
        ).model_dump(mode="json")


def _resolve_database_evidence(database_evidence, request_state):
    if database_evidence is None:
        latest = request_state.latest_database_evidence
        if latest is None:
            return None
        return request_state.database_evidence_artifacts.get(latest.evidence_id, latest)
    if isinstance(database_evidence, str):
        evidence_ref = database_evidence.strip()
        if evidence_ref in {"latest", "latest_database_evidence", "current"}:
            return _resolve_database_evidence(None, request_state)
        if evidence_ref.startswith("evidence:"):
            evidence_ref = evidence_ref.split(":", 1)[1]
        resolved = request_state.database_evidence_artifacts.get(evidence_ref)
        if resolved is None:
            raise ValueError(f"Forecast could not resolve database_evidence reference: {database_evidence}")
        return resolved
    if isinstance(database_evidence, dict):
        evidence_id = database_evidence.get("evidence_id")
        if evidence_id:
            return request_state.database_evidence_artifacts.get(evidence_id) or DatabaseEvidence.model_validate(database_evidence)
        return DatabaseEvidence.model_validate(database_evidence)
    return request_state.database_evidence_artifacts.get(database_evidence.evidence_id, database_evidence)

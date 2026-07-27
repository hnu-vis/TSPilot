"""Forecast tool placeholder."""
from __future__ import annotations

from datetime import datetime, timedelta
from math import ceil
import re
from statistics import median
from typing import Any

from pydantic import BaseModel, Field
from pydantic import field_validator

from core.timeseries.evidence_resolution import resolve_database_evidence
from core.timeseries.forecast_registry import default_forecast_model_name, get_forecast_model
from core.timeseries.normalization import normalize_timeseries_evidence
from schemas.database import DatabaseEvidence
from schemas.timeseries import ForecastPlan, ForecastResult, TimeSeriesSeries
from schemas.visualization import VisualizationPayload
from tools.base import BaseTool


DEFAULT_MAX_DIRECT_STEPS = 48
_DURATION_UNIT_SECONDS = {
    "s": 1,
    "sec": 1,
    "second": 1,
    "seconds": 1,
    "秒": 1,
    "m": 60,
    "min": 60,
    "minute": 60,
    "minutes": 60,
    "分钟": 60,
    "h": 3600,
    "hr": 3600,
    "hour": 3600,
    "hours": 3600,
    "小时": 3600,
    "d": 86400,
    "day": 86400,
    "days": 86400,
    "天": 86400,
    "日": 86400,
    "w": 604800,
    "week": 604800,
    "weeks": 604800,
    "周": 604800,
    "mo": 2592000,
    "month": 2592000,
    "months": 2592000,
    "月": 2592000,
    "y": 31536000,
    "yr": 31536000,
    "year": 31536000,
    "years": 31536000,
    "年": 31536000,
}
_DURATION_UNIT_PATTERN = "|".join(
    sorted((re.escape(unit) for unit in _DURATION_UNIT_SECONDS), key=len, reverse=True)
)
_DURATION_RE = re.compile(rf"(\d+(?:\.\d+)?)\s*({_DURATION_UNIT_PATTERN})", re.IGNORECASE)


class ForecastInput(BaseModel):
    database_evidence: DatabaseEvidence | dict | str | None = None
    horizon: int | str | dict | None = None
    model_name: str | None = None
    series_name: str | None = None
    constraints: dict | None = Field(default_factory=dict)

    @field_validator("horizon", mode="before")
    @classmethod
    def normalize_horizon(cls, value):
        if isinstance(value, dict):
            for key in ("steps", "horizon", "points", "count"):
                if key in value:
                    return cls.normalize_horizon(value[key])
            return value
        if isinstance(value, str):
            if _parse_duration_seconds(value) is not None:
                return value
            match = re.search(r"\d+", value)
            if match:
                return int(match.group(0))
        return value


class ForecastTool(BaseTool):
    async def execute(self, validated_input: ForecastInput, **kwargs) -> dict:
        request_state = kwargs.get("request_state")
        database_evidence = validated_input.database_evidence
        if request_state is not None:
            database_evidence = resolve_database_evidence(database_evidence, request_state, tool_label="Forecast")
        if database_evidence is None:
            raise ValueError("Forecast requires database_evidence or a latest_database_evidence in request state.")
        constraints = validated_input.constraints or {}
        preferred_series = validated_input.series_name or constraints.get("series_name")
        series = normalize_timeseries_evidence(
            database_evidence,
            series_name=preferred_series,
            value_field=preferred_series,
        )
        quality = _validate_forecast_evidence_quality(database_evidence, series, request_state, constraints)
        forecast_plan = _resolve_forecast_plan(validated_input, constraints, series)
        horizon = forecast_plan.requested_steps
        model_name = validated_input.model_name or constraints.get("model_name") or default_forecast_model_name()
        model = get_forecast_model(model_name)
        if forecast_plan.mode == "requires_rolling":
            visualization = _forecast_visualization(
                database_evidence=database_evidence,
                series=series,
                forecast_points=[],
                horizon=horizon,
                summary=(
                    f"{series.value_field} requires rolling forecast for {horizon} points; "
                    f"direct forecast is capped at {forecast_plan.max_direct_steps} points."
                ),
            )
            diagnostics = {
                "series_name": series.series_name,
                "model_registry_name": model.name,
                "horizon": horizon,
                "forecast_plan": forecast_plan.model_dump(mode="json"),
                "input_quality": quality,
            }
            return ForecastResult(
                forecast_id=f"forecast_{database_evidence.evidence_id}",
                model_name=model.name,
                horizon=horizon,
                status="requires_rolling",
                forecast_plan=forecast_plan,
                forecast_points=[],
                confidence_interval=[],
                diagnostics=diagnostics,
                visualizations=[visualization],
            ).model_dump(mode="json")

        model_output = model.forecast(series, horizon=horizon, params=constraints)
        forecast_points = model_output.forecast_points
        visualization = _forecast_visualization(
            database_evidence=database_evidence,
            series=series,
            forecast_points=forecast_points,
            horizon=horizon,
            summary=f"{series.value_field} 的未来 {horizon} 个点预测。",
        )
        return ForecastResult(
            forecast_id=f"forecast_{database_evidence.evidence_id}",
            model_name=model.name,
            horizon=horizon,
            status="succeeded",
            forecast_plan=forecast_plan,
            forecast_points=forecast_points,
            confidence_interval=model_output.confidence_interval,
            diagnostics={
                **model_output.diagnostics,
                "series_name": series.series_name,
                "model_registry_name": model.name,
                "horizon": horizon,
                "forecast_plan": forecast_plan.model_dump(mode="json"),
                "input_quality": quality,
            },
            visualizations=[visualization],
        ).model_dump(mode="json")


def _forecast_visualization(
    *,
    database_evidence: DatabaseEvidence,
    series: TimeSeriesSeries,
    forecast_points: list,
    horizon: int,
    summary: str,
) -> VisualizationPayload:
    return VisualizationPayload(
        visualization_id=f"viz_forecast_{database_evidence.evidence_id}",
        visualization_type="chart",
        visualization_kind="line",
        renderer="linechart",
        title=f"{series.value_field} forecast",
        summary=summary,
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
        requested_capabilities=["forecast"],
        time_column=series.time_field,
        primary_measure=series.value_field,
        display_priority=2,
    )


def _resolve_forecast_plan(
    validated_input: ForecastInput,
    constraints: dict,
    series: TimeSeriesSeries,
) -> ForecastPlan:
    sampling_interval = _infer_sampling_interval_seconds(series)
    duration_seconds = _duration_seconds_from_inputs(validated_input.horizon, constraints)
    explicit_steps = _horizon_steps_from_value(validated_input.horizon)
    if explicit_steps is None:
        explicit_steps = _horizon_steps_from_value(constraints.get("horizon"))

    if explicit_steps is not None:
        requested_steps = explicit_steps
        horizon_source = "explicit_steps"
        duration_seconds = None
    elif duration_seconds is not None:
        if sampling_interval is None:
            raise ValueError("Forecast duration requires at least two parseable timestamps to infer the sampling interval.")
        requested_steps = max(1, ceil(duration_seconds / sampling_interval))
        horizon_source = "duration_from_user"
    else:
        requested_steps = max(3, min(24, ceil(len(series.points) * 0.1)))
        horizon_source = "inferred_short_term_default"

    max_direct_steps = _positive_int(constraints.get("max_direct_steps"), DEFAULT_MAX_DIRECT_STEPS)
    forecast_start, forecast_end = _forecast_bounds(series, sampling_interval, requested_steps)
    if requested_steps > max_direct_steps:
        return ForecastPlan(
            mode="requires_rolling",
            horizon_source=horizon_source,
            requested_steps=requested_steps,
            resolved_steps=requested_steps,
            sampling_interval_seconds=sampling_interval,
            forecast_duration_seconds=duration_seconds,
            forecast_start=forecast_start,
            forecast_end=forecast_end,
            max_direct_steps=max_direct_steps,
            recommended_chunk_steps=max_direct_steps,
            reason=(
                f"Requested horizon {requested_steps} exceeds max direct forecast "
                f"window {max_direct_steps}; use rolling forecast chunks."
            ),
        )

    return ForecastPlan(
        mode="direct",
        horizon_source=horizon_source,
        requested_steps=requested_steps,
        resolved_steps=requested_steps,
        sampling_interval_seconds=sampling_interval,
        forecast_duration_seconds=duration_seconds,
        forecast_start=forecast_start,
        forecast_end=forecast_end,
        max_direct_steps=max_direct_steps,
        recommended_chunk_steps=None,
        reason=None,
    )


def _horizon_steps_from_value(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return max(1, value)
    if isinstance(value, float):
        return max(1, int(value))
    if isinstance(value, dict):
        for key in ("steps", "horizon", "points", "count"):
            steps = _horizon_steps_from_value(value.get(key))
            if steps is not None:
                return steps
        return None
    if isinstance(value, str):
        if _parse_duration_seconds(value) is not None:
            return None
        match = re.search(r"\d+", value)
        if match:
            return max(1, int(match.group(0)))
    return None


def _duration_seconds_from_inputs(horizon: Any, constraints: dict) -> int | None:
    for value in (
        horizon,
        constraints.get("forecast_duration"),
        constraints.get("horizon_duration"),
        constraints.get("duration"),
        constraints.get("forecast_window"),
        constraints.get("period"),
    ):
        seconds = _parse_duration_seconds(value)
        if seconds is not None:
            return seconds
    return None


def _parse_duration_seconds(value: Any) -> int | None:
    if isinstance(value, dict):
        for key in ("duration", "window", "period", "forecast_duration", "horizon_duration"):
            seconds = _parse_duration_seconds(value.get(key))
            if seconds is not None:
                return seconds
        return None
    if not isinstance(value, str):
        return None
    total = 0.0
    for amount, unit in _DURATION_RE.findall(value):
        total += float(amount) * _DURATION_UNIT_SECONDS[unit.lower()]
    if total <= 0:
        return None
    return max(1, int(total))


def _infer_sampling_interval_seconds(series: TimeSeriesSeries) -> int | None:
    parsed = [_parse_timestamp(point.timestamp) for point in series.points]
    parsed = [item for item in parsed if item is not None]
    if len(parsed) < 2:
        return None
    deltas = [
        int((current - previous).total_seconds())
        for previous, current in zip(parsed, parsed[1:])
        if current > previous
    ]
    if not deltas:
        return None
    return max(1, int(median(deltas)))


def _forecast_bounds(
    series: TimeSeriesSeries,
    sampling_interval_seconds: int | None,
    requested_steps: int,
) -> tuple[str | None, str | None]:
    if sampling_interval_seconds is None:
        return None, None
    last_timestamp = _parse_timestamp(series.points[-1].timestamp)
    if last_timestamp is None:
        return None, None
    forecast_start = last_timestamp + timedelta(seconds=sampling_interval_seconds)
    forecast_end = last_timestamp + timedelta(seconds=sampling_interval_seconds * requested_steps)
    return forecast_start.isoformat(), forecast_end.isoformat()


def _validate_forecast_evidence_quality(
    evidence: DatabaseEvidence,
    series: TimeSeriesSeries,
    request_state,
    constraints: dict,
) -> dict:
    query = (evidence.query or "").lower()
    raw_limit = "limit(" in query and "aggregatewindow" not in query
    coverage_start = series.points[0].timestamp
    coverage_end = series.points[-1].timestamp
    sampling_interval = _infer_sampling_interval_seconds(series)
    requested_end = _requested_time_range_end(request_state, constraints)
    covers_requested_range = True
    if requested_end is not None:
        observed_end = _parse_timestamp(coverage_end)
        tolerance_seconds = sampling_interval or 0
        covers_requested_range = bool(
            observed_end is not None
            and observed_end + timedelta(seconds=tolerance_seconds) >= requested_end
        )
    quality = {
        "coverage_start": coverage_start,
        "coverage_end": coverage_end,
        "sampling_interval_seconds": sampling_interval,
        "raw_limit_query": raw_limit,
        "covers_requested_range": covers_requested_range,
        "requested_range_end": requested_end.isoformat() if requested_end is not None else None,
    }
    if raw_limit and requested_end is not None and not covers_requested_range:
        raise ValueError(
            "Forecast evidence is incomplete for the requested time range because the query uses a raw limit. "
            "Use full-range aggregation or representative downsampling before forecasting."
        )
    return quality


def _requested_time_range_end(request_state, constraints: dict) -> datetime | None:
    for candidate in (
        constraints.get("time_range"),
        getattr(request_state, "time_range", None) if request_state is not None else None,
    ):
        if not isinstance(candidate, dict):
            continue
        parsed = _parse_timestamp(candidate.get("end") or candidate.get("to") or candidate.get("stop"))
        if parsed is not None:
            return parsed
    return None


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    normalized = value.strip().replace("Z", "+00:00")
    if "T" not in normalized and " " in normalized:
        normalized = normalized.replace(" ", "T")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _positive_int(value: Any, default: int) -> int:
    if isinstance(value, bool) or value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)

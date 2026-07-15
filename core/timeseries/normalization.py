"""Normalize database evidence into canonical time-series shapes."""
from __future__ import annotations

from datetime import datetime

from schemas.database import DatabaseEvidence
from schemas.timeseries import TimeSeriesPoint, TimeSeriesSeries


def normalize_timeseries_evidence(
    evidence: DatabaseEvidence,
    *,
    series_name: str | None = None,
    value_field: str | None = None,
) -> TimeSeriesSeries:
    if evidence.result_type != "timeseries":
        raise ValueError("Only timeseries evidence can be normalized for time-series tools.")

    data = evidence.data
    series_payload = data.get("series")
    preferred = series_name or value_field or data.get("value_field")
    if isinstance(series_payload, list) and series_payload:
        selected = None
        for item in series_payload:
            if not isinstance(item, dict):
                continue
            if preferred and preferred in {item.get("series_name"), item.get("value_field")}:
                selected = item
                break
        if selected is None:
            selected = series_payload[0]
        return _normalize_series_payload(selected)

    points = data.get("points")
    if not isinstance(points, list) or len(points) < 2:
        raise ValueError("Time-series evidence must contain at least two points.")

    normalized_points: list[TimeSeriesPoint] = []
    for point in points:
        if not isinstance(point, dict):
            continue
        timestamp = str(point.get("timestamp"))
        value = point.get("value")
        if value is None:
            continue
        normalized_points.append(
            TimeSeriesPoint(
                timestamp=_normalize_timestamp(timestamp),
                value=float(value),
            )
        )

    if len(normalized_points) < 2:
        raise ValueError("Time-series evidence must contain at least two numeric points.")

    return TimeSeriesSeries(
        series_name=data.get("series_name"),
        time_field=data.get("time_field", "timestamp"),
        value_field=data.get("value_field", "value"),
        points=normalized_points,
        labels=data.get("labels", {}),
    )


def _normalize_series_payload(payload: dict) -> TimeSeriesSeries:
    points = payload.get("points")
    if not isinstance(points, list) or len(points) < 2:
        raise ValueError("Selected series must contain at least two points.")
    normalized_points: list[TimeSeriesPoint] = []
    for point in points:
        if not isinstance(point, dict):
            continue
        timestamp = str(point.get("timestamp"))
        value = point.get("value")
        if value is None:
            continue
        normalized_points.append(
            TimeSeriesPoint(
                timestamp=_normalize_timestamp(timestamp),
                value=float(value),
            )
        )
    if len(normalized_points) < 2:
        raise ValueError("Selected series must contain at least two numeric points.")
    return TimeSeriesSeries(
        series_name=payload.get("series_name"),
        time_field=payload.get("time_field", "timestamp"),
        value_field=payload.get("value_field", "value"),
        points=normalized_points,
        labels=payload.get("labels", {}),
    )


def _normalize_timestamp(value: str) -> str:
    normalized = value.strip().replace("Z", "+00:00")
    if "T" not in normalized and " " in normalized:
        normalized = normalized.replace(" ", "T")
    return datetime.fromisoformat(normalized).isoformat()

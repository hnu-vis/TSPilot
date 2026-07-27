"""Simple forecast adapter for the first full backend slice."""
from __future__ import annotations

from datetime import datetime, timedelta
from statistics import median

import numpy as np

from schemas.timeseries import TimeSeriesPoint, TimeSeriesSeries


def linear_forecast(series: TimeSeriesSeries, horizon: int) -> list[TimeSeriesPoint]:
    values = np.array([point.value for point in series.points], dtype=float)
    x = np.arange(len(values), dtype=float)
    slope, intercept = np.polyfit(x, values, 1)
    horizon = max(1, horizon)
    future_points: list[TimeSeriesPoint] = []
    step_seconds = _infer_step_seconds(series)
    last_timestamp = _parse_timestamp(series.points[-1].timestamp)
    for index in range(1, horizon + 1):
        predicted = float(intercept + slope * (len(values) - 1 + index))
        future_timestamp = last_timestamp + timedelta(seconds=step_seconds * index)
        future_points.append(
            TimeSeriesPoint(
                timestamp=future_timestamp.isoformat(),
                value=predicted,
            )
        )
    return future_points


def _infer_step_seconds(series: TimeSeriesSeries) -> int:
    if len(series.points) < 2:
        return 60
    parsed = [_parse_timestamp(point.timestamp) for point in series.points]
    deltas = [
        int((current - previous).total_seconds())
        for previous, current in zip(parsed, parsed[1:])
        if current > previous
    ]
    if not deltas:
        return 60
    return max(1, int(median(deltas)))


def _parse_timestamp(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    if "T" not in normalized and " " in normalized:
        normalized = normalized.replace(" ", "T")
    return datetime.fromisoformat(normalized)

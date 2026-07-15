"""Simple forecast adapter for the first full backend slice."""
from __future__ import annotations

import numpy as np

from schemas.timeseries import TimeSeriesPoint, TimeSeriesSeries


def linear_forecast(series: TimeSeriesSeries, horizon: int) -> list[TimeSeriesPoint]:
    values = np.array([point.value for point in series.points], dtype=float)
    x = np.arange(len(values), dtype=float)
    slope, intercept = np.polyfit(x, values, 1)
    horizon = max(1, horizon)
    future_points: list[TimeSeriesPoint] = []
    step_seconds = _infer_step_seconds(series)
    last_timestamp = np.datetime64(series.points[-1].timestamp)
    for index in range(1, horizon + 1):
        predicted = float(intercept + slope * (len(values) - 1 + index))
        future_timestamp = last_timestamp + np.timedelta64(step_seconds * index, "s")
        future_points.append(
            TimeSeriesPoint(
                timestamp=str(future_timestamp).replace(" ", "T"),
                value=predicted,
            )
        )
    return future_points


def _infer_step_seconds(series: TimeSeriesSeries) -> int:
    if len(series.points) < 2:
        return 60
    first = np.datetime64(series.points[0].timestamp)
    second = np.datetime64(series.points[1].timestamp)
    delta = int((second - first) / np.timedelta64(1, "s"))
    return max(1, delta)


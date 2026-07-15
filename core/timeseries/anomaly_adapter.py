"""Simple anomaly adapter for the first full backend slice."""
from __future__ import annotations

import statistics

from schemas.timeseries import TimeSeriesSeries


def detect_zscore_anomalies(series: TimeSeriesSeries, threshold: float = 2.5) -> tuple[list[dict], list[dict]]:
    values = [point.value for point in series.points]
    if len(values) < 3:
        return [], []
    mean = statistics.fmean(values)
    stdev = statistics.pstdev(values)
    if stdev == 0:
        return [], []

    anomaly_points: list[dict] = []
    scores: list[dict] = []
    for point in series.points:
        score = (point.value - mean) / stdev
        scores.append({"timestamp": point.timestamp, "score": score})
        if abs(score) >= threshold:
            anomaly_points.append(
                {
                    "timestamp": point.timestamp,
                    "value": point.value,
                    "score": score,
                }
            )
    return anomaly_points, scores


"""Deterministic fact-family helpers for insight."""
from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict

from schemas.database import DatabaseEvidence

CANONICAL_FACT_TYPES = (
    "aggregation",
    "extreme",
    "trend",
    "difference",
    "rank",
    "distribution",
    "association",
    "outlier",
    "forecast",
    "seasonality",
    "proportion",
    "categorization",
)

FACT_TYPE_ALIASES = {
    "aggregate": "aggregation",
    "aggregation": "aggregation",
    "avg": "aggregation",
    "mean": "aggregation",
    "sum": "aggregation",
    "count": "aggregation",
    "extreme": "extreme",
    "extremes": "extreme",
    "extrema": "extreme",
    "max": "extreme",
    "min": "extreme",
    "trend": "trend",
    "difference": "difference",
    "comparison": "difference",
    "change": "difference",
    "change_percent": "difference",
    "delta": "difference",
    "rank": "rank",
    "ranking": "rank",
    "distribution": "distribution",
    "pattern": "distribution",
    "association": "association",
    "correlation": "association",
    "outlier": "outlier",
    "outliers": "outlier",
    "anomaly": "outlier",
    "anomalies": "outlier",
    "forecast": "forecast",
    "forecasts": "forecast",
    "prediction": "forecast",
    "predictions": "forecast",
    "predict": "forecast",
    "seasonality": "seasonality",
    "seasonal": "seasonality",
    "periodicity": "seasonality",
    "proportion": "proportion",
    "ratio": "proportion",
    "share": "proportion",
    "categorization": "categorization",
    "category": "categorization",
    "bucket": "categorization",
}

DEFAULT_FACT_TYPES = ("trend", "difference", "extreme")


def canonicalize_fact_type(fact_type: str | None) -> str | None:
    normalized = str(fact_type or "").strip().lower()
    if not normalized:
        return None
    return FACT_TYPE_ALIASES.get(normalized)


def normalize_requested_fact_types(requested_fact_types: list[str], *, allow_default: bool = True) -> list[str]:
    normalized: list[str] = []
    requested = requested_fact_types or (list(DEFAULT_FACT_TYPES) if allow_default else [])
    for item in requested:
        canonical = canonicalize_fact_type(item)
        if canonical and canonical not in normalized:
            normalized.append(canonical)
    if normalized:
        return normalized
    return list(DEFAULT_FACT_TYPES) if allow_default else []


def evidence_rows(evidence: DatabaseEvidence) -> tuple[list[dict], list[str], str, str]:
    data = evidence.data or {}
    time_field = str(data.get("time_field") or "timestamp")
    value_field = str(data.get("value_field") or "value")

    rows_payload = data.get("rows")
    if isinstance(rows_payload, list) and rows_payload:
        rows = [dict(row) for row in rows_payload if isinstance(row, dict)]
    else:
        rows = []
        for point in data.get("points", []) or []:
            if not isinstance(point, dict):
                continue
            row = {time_field: point.get("timestamp"), value_field: point.get("value")}
            for key, value in point.items():
                if key not in {"timestamp", "value"}:
                    row[key] = value
            labels = data.get("labels") or {}
            if isinstance(labels, dict):
                for key, value in labels.items():
                    row.setdefault(key, value)
            rows.append(row)

    columns = list(evidence.columns or [])
    if not columns and rows:
        seen: list[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        columns = seen
    return rows, columns, time_field, value_field


def numeric_columns(rows: list[dict], columns: list[str]) -> dict[str, list[tuple[int, float]]]:
    numeric: dict[str, list[tuple[int, float]]] = {}
    for column in columns:
        if _is_time_column(column):
            continue
        values: list[tuple[int, float]] = []
        for index, row in enumerate(rows):
            number = to_number(row.get(column))
            if number is not None and math.isfinite(number):
                values.append((index, number))
        if len(values) >= 2:
            numeric[column] = values
    return numeric


def categorical_columns(rows: list[dict], columns: list[str], numeric: dict[str, list[tuple[int, float]]]) -> list[str]:
    categories: list[str] = []
    for column in columns:
        if column in numeric or _is_time_column(column):
            continue
        values = [row.get(column) for row in rows if row.get(column) not in (None, "")]
        unique = {str(value) for value in values}
        if len(unique) >= 2:
            categories.append(column)
    return categories


def to_number(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def linear_slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = sum((x - x_mean) ** 2 for x in xs)
    return numerator / denominator if denominator else 0.0


def quantile(sorted_values: list[float], ratio: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = ratio * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    denominator = math.sqrt(x_var * y_var)
    return numerator / denominator if denominator else None


def detect_period(values: list[float]) -> tuple[int | None, float]:
    max_corr = 0.0
    best_period: int | None = None
    upper = min(len(values) // 2, 96)
    for period in range(2, upper + 1):
        corr = pearson(values[:-period], values[period:])
        if corr is not None and corr > max_corr:
            max_corr = corr
            best_period = period
    return (best_period, max_corr) if max_corr > 0.3 else (None, max_corr)


def seasonal_strength(values: list[float], period: int) -> float:
    if len(values) < 2 * period:
        return 0.0
    buckets = []
    for offset in range(period):
        items = [values[index] for index in range(offset, len(values), period)]
        if items:
            buckets.append(statistics.fmean(items))
    total_var = population_variance(values)
    return population_variance(buckets) / total_var if total_var > 0 else 0.0


def population_variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.fmean(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def series_count(evidence: DatabaseEvidence) -> int:
    data = evidence.data or {}
    series_payload = data.get("series")
    if isinstance(series_payload, list) and series_payload:
        return len(series_payload)
    rows, columns, _, _ = evidence_rows(evidence)
    numeric = numeric_columns(rows, columns)
    categories = categorical_columns(rows, columns, numeric)
    if categories:
        category = categories[0]
        unique = {str(row.get(category)) for row in rows if row.get(category) not in (None, "")}
        if len(unique) >= 2:
            return len(unique)
    return 1


def supports_multi_series_runtime(evidence: DatabaseEvidence) -> bool:
    data = evidence.data or {}
    if isinstance(data.get("series"), list) and data.get("series"):
        return True
    rows, columns, _, _ = evidence_rows(evidence)
    numeric = numeric_columns(rows, columns)
    categories = categorical_columns(rows, columns, numeric)
    return len(categories) >= 1 and len(numeric) >= 1


def _is_time_column(column: str) -> bool:
    normalized = str(column or "").lower()
    return normalized in {"time", "timestamp", "_time"} or "time" in normalized or "date" in normalized


def best_category_column(rows: list[dict], columns: list[str], numeric: dict[str, list[tuple[int, float]]]) -> str | None:
    categories = categorical_columns(rows, columns, numeric)
    return categories[0] if categories else None


def grouped_numeric_values(
    rows: list[dict],
    category_column: str,
    measure_column: str,
) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        category = row.get(category_column)
        value = to_number(row.get(measure_column))
        if category not in (None, "") and value is not None:
            grouped[str(category)].append(value)
    return grouped


def most_common_category(rows: list[dict], category_column: str) -> tuple[str, int] | None:
    counts = Counter(str(row.get(category_column)) for row in rows if row.get(category_column) not in (None, ""))
    if not counts:
        return None
    return counts.most_common(1)[0]

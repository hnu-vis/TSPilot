"""Canonical inputs for generated time-series analysis code."""
from __future__ import annotations

from datetime import datetime
from typing import Any


TIME_CANDIDATES = ("timestamp", "time", "_time", "date", "datetime")
VALUE_CANDIDATES = ("value", "_value", "price", "close", "amount")


def build_canonical_analysis_context(
    *,
    rows: list[dict],
    points: list[dict],
    columns: list[str],
    metadata: dict | None = None,
    diagnostics: dict | None = None,
) -> dict:
    """Build a stable, promptable data plane without computing user metrics."""

    row_items = [dict(item) for item in rows if isinstance(item, dict)]
    point_items = [dict(item) for item in points if isinstance(item, dict)]
    # Rows retain dimensions that the primary points projection may omit for
    # multi-series evidence. Prefer them whenever they are available.
    source_items = row_items or point_items
    time_col = _first_key(source_items, [*TIME_CANDIDATES, *columns])
    value_col = _first_numeric_key(source_items, [*VALUE_CANDIDATES, *columns])
    series = _canonical_series(source_items, time_col, value_col)
    dimension_cols = _dimension_keys(source_items, time_col=time_col, value_col=value_col)
    return {
        "schema": {
            "columns": list(columns),
            "time_col": time_col,
            "value_col": value_col,
            "row_count": len(row_items),
            "point_count": len(point_items),
            "series_count": len(series),
            "dimension_cols": dimension_cols,
            "sample_rows": row_items[:5],
            "sample_points": point_items[:5],
            "sample_series": series[:5],
        },
        "variables": {
            "df": "pandas.DataFrame built from canonical series when pandas is available, else None",
            "series": "list[dict] with timestamp/value plus every original row dimension",
            "time": "df[time_col] when available",
            "value": "df[value_col] when available",
            "time_col": time_col,
            "value_col": value_col,
            "rows": "original row records",
            "points": "original point records",
            "columns": "original column names",
            "metadata": "database evidence metadata",
            "diagnostics": "database evidence diagnostics",
        },
        "metadata": dict(metadata or {}),
        "diagnostics": dict(diagnostics or {}),
    }


def canonical_namespace_values(payload: dict) -> dict:
    """Return runtime variables injected into the sandbox namespace."""

    rows = [dict(row) for row in payload.get("rows") or [] if isinstance(row, dict)]
    points = [dict(point) for point in payload.get("points") or [] if isinstance(point, dict)]
    columns = list(payload.get("columns") or [])
    context = build_canonical_analysis_context(
        rows=rows,
        points=points,
        columns=columns,
        metadata=dict(payload.get("metadata") or {}),
        diagnostics=dict(payload.get("diagnostics") or {}),
    )
    schema = context["schema"]
    series = list(schema.get("sample_series") or [])
    # Rebuild full series; the schema sample is intentionally bounded for diagnostics.
    source_items = rows or points
    series = _canonical_series(source_items, schema.get("time_col"), schema.get("value_col"))
    df = None
    time = None
    value = None
    try:
        import pandas as pd  # type: ignore

        df = pd.DataFrame(series if series else source_items)
        time_col = schema.get("time_col")
        value_col = schema.get("value_col")
        if time_col and time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
            df = df.sort_values(time_col).reset_index(drop=True)
            time = df[time_col]
        if value_col and value_col in df.columns:
            df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
            value = df[value_col]
    except Exception:
        df = None
        time = None
        value = None
    return {
        "analysis_context": context,
        "series": series,
        "df": df,
        "time": time,
        "value": value,
        "time_col": schema.get("time_col"),
        "value_col": schema.get("value_col"),
    }


def _canonical_series(items: list[dict], time_col: str | None, value_col: str | None) -> list[dict]:
    series: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        value = _as_float(item.get(value_col)) if value_col else None
        if value is None:
            continue
        normalized = dict(item)
        normalized["timestamp"] = item.get(time_col) if time_col else item.get("timestamp")
        normalized["value"] = value
        series.append(normalized)
    if time_col:
        series.sort(key=lambda row: _time_sort_key(row.get("timestamp")))
    return series


def _first_key(items: list[dict], candidates: list[str] | tuple[str, ...]) -> str | None:
    available = {str(key) for item in items[:20] for key in item.keys()}
    for candidate in candidates:
        if candidate in available:
            return str(candidate)
    return None


def _first_numeric_key(items: list[dict], candidates: list[str] | tuple[str, ...]) -> str | None:
    available = {str(key) for item in items[:20] for key in item.keys()}
    ordered = [str(candidate) for candidate in candidates if str(candidate) in available]
    ordered.extend(sorted(available - set(ordered)))
    for candidate in ordered:
        if candidate in TIME_CANDIDATES:
            continue
        values = [_as_float(item.get(candidate)) for item in items[:50] if isinstance(item, dict)]
        numeric = [value for value in values if value is not None]
        if numeric:
            return candidate
    return None


def _dimension_keys(items: list[dict], *, time_col: str | None, value_col: str | None) -> list[str]:
    """Return non-numeric columns that can distinguish time-series groups."""
    excluded = {key for key in (time_col, value_col, "timestamp", "value") if key}
    available = sorted({str(key) for item in items[:50] for key in item if str(key) not in excluded})
    dimensions: list[str] = []
    for key in available:
        values = [item.get(key) for item in items[:50] if item.get(key) not in (None, "")]
        if values and not all(_as_float(value) is not None for value in values):
            dimensions.append(key)
    return dimensions


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _time_sort_key(value: Any) -> tuple[int, str]:
    if value in (None, ""):
        return (1, "")
    text = str(value)
    try:
        normalized = text.replace("Z", "+00:00")
        return (0, datetime.fromisoformat(normalized).isoformat())
    except ValueError:
        return (0, text)

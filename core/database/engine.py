"""Deterministic execution helpers for database-backed queries."""
from __future__ import annotations

from pathlib import Path

from .connector import QueryResult
from .repair import classify_query_error, should_retry_query
from core.intent import AGGREGATION_FACT_TYPES, build_intent_profile_fallback
from schemas.database import DatabaseEvidence


async def execute_query(connector, query: str, *, timeout: int | None = None) -> QueryResult:
    attempts = 0
    while True:
        try:
            return await connector.execute(query, timeout=timeout)
        except Exception as exc:
            repair = classify_query_error(exc)
            if not should_retry_query(repair, attempts):
                raise RuntimeError(repair["message"]) from exc
            attempts += 1


async def execute_range_query(connector, query: str, *, start, end, step: str) -> QueryResult:
    attempts = 0
    while True:
        try:
            return await connector.get_range(query, start, end, step=step)
        except Exception as exc:
            repair = classify_query_error(exc)
            if not should_retry_query(repair, attempts):
                raise RuntimeError(repair["message"]) from exc
            attempts += 1


def infer_evidence_family(message: str) -> str:
    normalized = message.lower()
    intent_profile = build_intent_profile_fallback(message)
    fact_types = set(intent_profile.get("requested_fact_types") or [])
    has_statistics_request = bool(fact_types & AGGREGATION_FACT_TYPES)
    has_timeseries_request = bool(fact_types & {"seasonality", "trend", "outlier", "forecast", "association", "difference"})

    metric_keywords = ("metric list", "metrics", "available metrics", "有哪些指标", "有哪些 metric")
    if any(keyword in normalized for keyword in metric_keywords):
        return "metric_list"

    schema_keywords = (
        "schema",
        "field",
        "label",
        "有哪些表",
        "有哪些字段",
        "结构",
        "labels",
        "schema preview",
    )
    if any(keyword in normalized for keyword in schema_keywords):
        return "schema"

    if has_statistics_request:
        return "statistics"

    if has_timeseries_request:
        return "timeseries"

    table_keywords = ("group by", "table", "rows", "明细", "列表")
    if any(keyword in normalized for keyword in table_keywords):
        return "table"

    return "timeseries"


def infer_prometheus_metric(message: str, schema) -> str | None:
    normalized = message.lower()
    metric_names = [table.name for table in schema.tables]
    for metric_name in metric_names:
        if metric_name.lower() in normalized:
            return metric_name
    return metric_names[0] if metric_names else None


def normalize_query_result(
    *,
    database_id: str,
    database_type: str,
    query_language: str,
    query: str,
    result,
) -> DatabaseEvidence:
    rows = list(getattr(result, "rows", []) or [])
    columns = _normalize_result_columns(list(getattr(result, "columns", []) or []))
    rows = [_normalize_result_row(dict(row)) for row in rows]
    result_diagnostics = _result_fidelity_diagnostics(result, len(rows))
    if not rows:
        return DatabaseEvidence(
            evidence_id=f"evi_{database_id}_empty",
            result_type="table",
            database=database_id,
            query_language=query_language,
            query=query,
            summary="The query completed but returned no rows.",
            data={"rows": []},
            columns=columns,
            metadata={"database_type": database_type},
            diagnostics=result_diagnostics,
        )

    if "timestamp" in columns and "value" in columns:
        category_columns = [
            column
            for column in columns
            if column not in {"timestamp", "value"} and not _is_numeric_column(rows, column)
        ]
        if category_columns:
            category = category_columns[0]
            grouped: dict[str, list[dict]] = {}
            for row in rows:
                label = row.get(category)
                if label in (None, "") or row.get("value") is None:
                    continue
                grouped.setdefault(str(label), []).append(
                    {"timestamp": str(row["timestamp"]), "value": float(row["value"])}
                )
            if grouped:
                labels = sorted(grouped)
                primary = labels[0]
                return DatabaseEvidence(
                    evidence_id=f"evi_{database_id}_{query.replace(' ', '_')}",
                    result_type="timeseries",
                    database=database_id,
                    query_language=query_language,
                    query=query,
                    summary=f"Loaded {len(rows)} rows across {len(labels)} series for query '{query}'.",
                    data={
                        "points": grouped[primary],
                        "rows": rows,
                        "series": [
                            {
                                "series_name": label,
                                "value_field": "value",
                                "time_field": "timestamp",
                                "points": grouped[label],
                                "labels": {category: label},
                            }
                            for label in labels
                        ],
                        "time_field": "timestamp",
                        "value_field": "value",
                        "series_name": primary,
                        "labels": {category: primary},
                    },
                    columns=columns,
                    metadata={"database_type": database_type},
                    diagnostics={
                        **result_diagnostics,
                        "series_count": len(labels),
                        "series_dimension": category,
                    },
                )
        points = [
            {"timestamp": str(row["timestamp"]), "value": float(row["value"])}
            for row in rows
            if row.get("value") is not None
        ]
        return DatabaseEvidence(
            evidence_id=f"evi_{database_id}_{query.replace(' ', '_')}",
            result_type="timeseries",
            database=database_id,
            query_language=query_language,
            query=query,
            summary=f"Loaded {len(points)} points for query '{query}'.",
            data={
                "points": points,
                "time_field": "timestamp",
                "value_field": "value",
                "series_name": query,
                "labels": {},
            },
            columns=["timestamp", "value"],
            metadata={"database_type": database_type},
            diagnostics=result_diagnostics,
        )

    numeric_columns = [
        column
        for column in columns
        if column != "timestamp" and _is_numeric_column(rows, column)
    ]
    if "timestamp" in columns and numeric_columns:
        primary = numeric_columns[0]
        sampled_rows = []
        primary_points = []
        series = []
        for column in numeric_columns:
            column_points = []
            for row in rows:
                value = row.get(column)
                if value is None:
                    continue
                numeric_value = float(value)
                if column == primary:
                    primary_points.append({"timestamp": str(row["timestamp"]), "value": numeric_value})
                column_points.append({"timestamp": str(row["timestamp"]), "value": numeric_value})
            series.append(
                {
                    "series_name": column,
                    "value_field": column,
                    "time_field": "timestamp",
                    "points": column_points,
                    "labels": {},
                }
            )
        for row in rows:
            sampled_rows.append({key: row.get(key) for key in ["timestamp", *numeric_columns] if key in row})
        return DatabaseEvidence(
            evidence_id=f"evi_{database_id}_{query.replace(' ', '_')}",
            result_type="timeseries",
            database=database_id,
            query_language=query_language,
            query=query,
            summary=f"Loaded {len(rows)} rows across {len(numeric_columns)} series for query '{query}'.",
            data={
                "points": primary_points,
                "rows": sampled_rows,
                "series": series,
                "time_field": "timestamp",
                "value_field": primary,
                "series_name": primary,
                "labels": {},
            },
            columns=["timestamp", *numeric_columns],
            metadata={"database_type": database_type},
            diagnostics={
                **result_diagnostics,
                "series_count": len(numeric_columns),
                "selected_fields": numeric_columns,
            },
        )

    return DatabaseEvidence(
        evidence_id=f"evi_{database_id}_table",
        result_type="table",
        database=database_id,
        query_language=query_language,
        query=query,
        summary=f"Loaded {len(rows)} rows.",
        data={"rows": rows},
        columns=columns,
        metadata={"database_type": database_type},
        diagnostics=result_diagnostics,
    )


def _result_fidelity_diagnostics(result, materialized_row_count: int) -> dict:
    row_count = getattr(result, "row_count", None)
    truncated = bool(getattr(result, "truncated", False))
    total_rows = row_count if isinstance(row_count, int) else materialized_row_count
    return {
        "row_count_total": total_rows,
        "row_count_materialized": materialized_row_count,
        "is_full_fidelity": not truncated and materialized_row_count == total_rows,
        "truncated": truncated,
        "sampling_policy": {
            "analysis_input": "query_result_rows",
            "prompt_preview": "runtime_prompt_safe_sampling",
        },
    }


def _normalize_result_columns(columns: list[str]) -> list[str]:
    normalized = []
    for column in columns:
        if column == "time":
            normalized.append("timestamp")
        else:
            normalized.append(column)
    return normalized


def _normalize_result_row(row: dict) -> dict:
    if "time" in row and "timestamp" not in row:
        row["timestamp"] = row.pop("time")
    return row


def build_reference_dataset_statistics_evidence(
    *,
    database_id: str,
    database_type: str,
    config_path: Path,
    dataset_path: Path,
    value_field: str,
    time_field: str,
    values: list[float],
) -> DatabaseEvidence:
    stats = {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "avg": sum(values) / len(values),
        "sum": sum(values),
    }
    return DatabaseEvidence(
        evidence_id=f"evi_{database_id}_{value_field}_stats",
        result_type="statistics",
        database=database_id,
        query_language="reference_dataset",
        query=f"reference_dataset:{value_field}:statistics",
        summary=f"Computed statistics for {value_field} over {len(values)} rows.",
        data={"statistics": stats, "value_field": value_field, "time_field": time_field},
        columns=["metric", "value"],
        metadata={
            "config_path": str(config_path),
            "dataset_path": str(dataset_path),
            "database_type": database_type,
        },
        diagnostics={"selected_field": value_field},
    )


def build_reference_dataset_timeseries_evidence(
    *,
    database_id: str,
    database_type: str,
    config_path: Path,
    dataset_path: Path,
    value_field: str,
    value_fields: list[str] | None,
    time_field: str,
    rows: list[dict] | None,
    points: list[dict],
    source: str,
) -> DatabaseEvidence:
    selected_fields = value_fields or [value_field]
    series = []
    if rows:
        for field in selected_fields:
            column_points = []
            for row in rows:
                value = row.get(field)
                if value is None:
                    continue
                column_points.append({"timestamp": str(row[time_field]), "value": float(value)})
            series.append(
                {
                    "series_name": field,
                    "value_field": field,
                    "time_field": time_field,
                    "points": column_points,
                    "labels": {"source": source},
                }
            )
    return DatabaseEvidence(
        evidence_id=f"evi_{database_id}_{value_field}",
        result_type="timeseries",
        database=database_id,
        query_language="reference_dataset",
        query=f"reference_dataset:{value_field}",
        summary=(
            f"Loaded {len(points)} points for {value_field} from the configured reference dataset."
            if len(selected_fields) == 1
            else f"Loaded {len(rows or [])} rows across {len(selected_fields)} series from the configured reference dataset."
        ),
        data={
            "points": points,
            "rows": rows or [],
            "series": series,
            "time_field": time_field,
            "value_field": value_field,
            "series_name": value_field,
            "labels": {"source": source},
        },
        columns=[time_field, *selected_fields],
        metadata={
            "config_path": str(config_path),
            "dataset_path": str(dataset_path),
            "database_type": database_type,
        },
        diagnostics={"selected_field": value_field, "selected_fields": selected_fields, "series_count": len(selected_fields)},
    )


def _is_numeric_column(rows: list[dict], column: str) -> bool:
    numeric_count = 0
    for row in rows:
        value = row.get(column)
        if value is None:
            continue
        try:
            float(value)
            numeric_count += 1
        except (TypeError, ValueError):
            return False
    return numeric_count >= 2

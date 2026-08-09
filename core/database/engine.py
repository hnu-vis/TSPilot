"""Execution helpers for database-backed queries."""
from __future__ import annotations

import hashlib

from .connector import QueryResult
from schemas.database import DatabaseEvidence


async def execute_query(connector, query: str, *, timeout: int | None = None) -> QueryResult:
    return await connector.execute(query, timeout=timeout)


def evidence_id_for_query(database_id: str, query: str, result_type: str) -> str:
    digest = hashlib.sha1(query.encode("utf-8")).hexdigest()[:12]
    safe_result_type = "".join(char if char.isalnum() else "_" for char in result_type).strip("_") or "query"
    return f"evi_{database_id}_{safe_result_type}_{digest}"


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
                    evidence_id=evidence_id_for_query(database_id, query, "timeseries"),
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
            evidence_id=evidence_id_for_query(database_id, query, "timeseries"),
            result_type="timeseries",
            database=database_id,
            query_language=query_language,
            query=query,
            summary=f"Loaded {len(points)} points for query '{query}'.",
            data={
                "points": points,
                "rows": rows,
                "time_field": "timestamp",
                "value_field": "value",
                "series_name": query,
                "labels": {},
            },
            columns=columns,
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
        return DatabaseEvidence(
            evidence_id=evidence_id_for_query(database_id, query, "timeseries"),
            result_type="timeseries",
            database=database_id,
            query_language=query_language,
            query=query,
            summary=f"Loaded {len(rows)} rows across {len(numeric_columns)} series for query '{query}'.",
            data={
                "points": primary_points,
                "rows": rows,
                "series": series,
                "time_field": "timestamp",
                "value_field": primary,
                "series_name": primary,
                "labels": {},
            },
            columns=columns,
            metadata={"database_type": database_type},
            diagnostics={
                **result_diagnostics,
                "series_count": len(numeric_columns),
                "selected_fields": numeric_columns,
            },
        )

    return DatabaseEvidence(
        evidence_id=evidence_id_for_query(database_id, query, "table"),
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

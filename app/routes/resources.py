"""Read-only resource endpoints for the frontend workspace."""
from __future__ import annotations

import csv
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.settings import get_settings
from core.database import DatabaseFactory, metric_list_preview, schema_preview

router = APIRouter(prefix="/api/v1/resources", tags=["resources"])


def _public_database_config(db_id: str, config: dict) -> dict:
    return {
        "id": db_id,
        "name": config.get("name") or db_id,
        "type": config.get("type") or config.get("db_type") or "unknown",
        "status": config.get("status", "unknown"),
        "host": config.get("host"),
        "port": config.get("port"),
        "database": config.get("database"),
        "display_name": config.get("display_name") or config.get("name") or db_id,
        "config_source": config.get("config_source"),
        "has_reference_dataset": bool(config.get("reference_dataset")),
    }


@router.get("/databases")
async def list_database_resources() -> dict:
    """List configured databases without exposing credentials."""
    await DatabaseFactory.load_databases()
    databases = [
        _public_database_config(db_id, config)
        for db_id, config in sorted(DatabaseFactory._databases.items())
    ]
    return {"databases": databases, "total": len(databases)}


@router.get("/databases/{database_id}/preview")
async def preview_database_resource(database_id: str) -> dict:
    """Return a lightweight schema or metric preview for one database."""
    config = await DatabaseFactory.get_database(database_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Database '{database_id}' was not found.")

    public_config = _public_database_config(database_id, config)
    reference_dataset = config.get("reference_dataset")
    if isinstance(reference_dataset, dict):
        preview = _reference_dataset_preview(config)
        return {
            "database": public_config,
            "preview_kind": "reference_dataset",
            "summary": f"Loaded reference dataset schema for {len(preview.get('tables_or_measurements', []))} object.",
            "preview": preview,
        }

    try:
        connector = await DatabaseFactory.create_connector(**dict(config))
        async with connector:
            schema = await connector.get_schema()
    except Exception as exc:
        return {
            "database": public_config,
            "preview_kind": "error",
            "summary": "Unable to load schema preview.",
            "error": str(exc),
        }

    db_type = str(public_config.get("type") or "").lower()
    if db_type == "prometheus":
        preview = metric_list_preview(schema)
        return {
            "database": public_config,
            "preview_kind": "metrics",
            "summary": f"Loaded {len(preview.get('metrics', []))} metrics.",
            "preview": preview,
        }

    preview = schema_preview(schema)
    return {
        "database": public_config,
        "preview_kind": "schema",
        "summary": f"Loaded {len(preview.get('tables_or_measurements', []))} schema objects.",
        "preview": preview,
    }


def _reference_dataset_preview(config: dict) -> dict:
    reference_dataset = config.get("reference_dataset") if isinstance(config.get("reference_dataset"), dict) else {}
    dataset_path = _resolve_reference_dataset_path(config, reference_dataset)
    sample_rows = _read_sample_rows(dataset_path, limit=3)
    row_count = _count_csv_rows(dataset_path)
    field_columns = reference_dataset.get("field_columns")
    if not isinstance(field_columns, list):
        value_column = reference_dataset.get("value_column")
        field_columns = [value_column] if value_column else []
    time_column = reference_dataset.get("timestamp_column")
    columns = []
    if time_column:
        columns.append({"name": str(time_column), "data_type": "datetime", "nullable": True})
    columns.extend(
        {"name": str(column), "data_type": "unknown", "nullable": True}
        for column in field_columns
        if column not in (None, "")
    )
    table_name = (
        reference_dataset.get("measurement")
        or reference_dataset.get("metric_name")
        or reference_dataset.get("table")
        or reference_dataset.get("series_name")
        or config.get("name")
    )
    table = {
        "name": table_name,
        "schema": "",
        "type": "reference_dataset",
        "row_count": row_count,
        "columns": columns,
        "field_values": [str(column) for column in field_columns if column not in (None, "")],
        "sample_rows": sample_rows,
    }
    return {
        "tables_or_measurements": [table],
        "fields": [
            {"table": table_name, **column}
            for column in columns
        ],
        "labels_or_tags": [],
        "time_columns": [str(time_column)] if time_column else [],
        "metadata": {
            "database_type": config.get("type") or config.get("db_type"),
            "reference_dataset": {
                "dataset_path": reference_dataset.get("dataset_path"),
                "row_count": row_count,
                "timestamp_column": time_column,
                "source": reference_dataset.get("source"),
            },
        },
    }


def _resolve_reference_dataset_path(config: dict, reference_dataset: dict) -> Path | None:
    raw_path = reference_dataset.get("resolved_dataset_path") or reference_dataset.get("dataset_path")
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    return (Path(get_settings().tspilot_root) / path).resolve()


def _count_csv_rows(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return sum(1 for _ in csv.DictReader(handle))
    except Exception:
        return None


def _read_sample_rows(path: Path | None, *, limit: int) -> list[dict]:
    if path is None or not path.exists() or limit <= 0:
        return []
    rows = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(dict(row))
                if len(rows) >= limit:
                    break
    except Exception:
        return []
    return rows


@router.get("/knowledge")
async def list_knowledge_resources() -> dict:
    """List the local knowledge resource exposed to the agent."""
    settings = get_settings()
    root = settings.resolved_knowledge_base_dir
    resources = []
    if root.exists():
        file_count = sum(
            1
            for path in root.rglob("*")
            if path.is_file() and not any(part.startswith(".") for part in path.parts)
        )
        resources.append(
            {
                "id": "local",
                "name": "Local knowledge",
                "type": "local",
                "status": "available",
                "root": str(root),
                "document_count": file_count,
            }
        )
    return {"knowledge": resources, "total": len(resources)}


@router.get("/model")
async def get_model_resource() -> dict:
    """Return the configured backend model label."""
    settings = get_settings()
    return {"model": settings.openai_model}

"""Read-only resource endpoints for the frontend workspace."""
from __future__ import annotations

import csv
from time import perf_counter
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.settings import get_settings
from core.database import DatabaseFactory, metric_list_preview, schema_preview
from core.data_fact.memory import prompt_fact_memory_view, read_fact_memory

router = APIRouter(prefix="/api/v1/resources", tags=["resources"])


class DatabaseConfigPayload(BaseModel):
    """Frontend editable database connection config."""

    name: str = Field(min_length=1)
    type: str = Field(min_length=1)
    host: str | None = None
    port: int | None = None
    database: str | None = None
    username: str | None = None
    password: str | None = None
    display_name: str | None = None
    ssl_enabled: bool | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class DatabaseConfigUpdatePayload(BaseModel):
    """Partial database connection config update."""

    name: str | None = Field(default=None, min_length=1)
    type: str | None = Field(default=None, min_length=1)
    host: str | None = None
    port: int | None = None
    database: str | None = None
    username: str | None = None
    password: str | None = None
    display_name: str | None = None
    ssl_enabled: bool | None = None
    extra: dict[str, Any] | None = None


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
        "username": config.get("username"),
        "ssl_enabled": bool(config.get("ssl_enabled", False)),
    }


@router.get("/databases")
async def list_database_resources() -> dict:
    """List configured databases without exposing credentials."""
    databases = await _list_public_databases()
    return {"databases": databases, "total": len(databases)}


@router.post("/databases", status_code=201)
async def create_database_resource(payload: DatabaseConfigPayload) -> dict:
    """Create a database connection config."""
    config = _editable_payload_to_config(payload.model_dump(exclude_unset=True))
    existing = await DatabaseFactory.get_database(str(config["name"]))
    if existing:
        raise HTTPException(status_code=409, detail=f"Database '{config['name']}' already exists.")
    db_id = await DatabaseFactory.add_database(**config)
    saved_config = await DatabaseFactory.get_database(db_id)
    profile_cache = DatabaseFactory.read_profile_cache(db_id)
    return {"database": _public_database_config(db_id, saved_config or config), "profile_cache": profile_cache}


@router.get("/databases/{database_id}")
async def get_database_resource(database_id: str) -> dict:
    """Return one configured database without exposing credentials."""
    config = await DatabaseFactory.get_database(database_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Database '{database_id}' was not found.")
    return {"database": _public_database_config(database_id, config)}


@router.patch("/databases/{database_id}")
async def update_database_resource(database_id: str, payload: DatabaseConfigUpdatePayload) -> dict:
    """Update a database connection config."""
    updates = _editable_payload_to_config(payload.model_dump(exclude_unset=True))
    saved_config = await DatabaseFactory.update_database(database_id, **updates)
    if not saved_config:
        raise HTTPException(status_code=404, detail=f"Database '{database_id}' was not found.")
    return {"database": _public_database_config(database_id, saved_config)}


@router.delete("/databases/{database_id}")
async def delete_database_resource(database_id: str) -> dict:
    """Delete a database connection config."""
    deleted = await DatabaseFactory.delete_database(database_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Database '{database_id}' was not found.")
    return {"deleted": True, "database_id": database_id}


@router.post("/databases/{database_id}/test")
async def test_database_resource(database_id: str) -> dict:
    """Test one configured database connection and persist its status."""
    config = await DatabaseFactory.get_database(database_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Database '{database_id}' was not found.")

    started = perf_counter()
    try:
        connector = await DatabaseFactory.create_connector(**dict(config))
        result = await connector.test_connection()
    except Exception as exc:
        result = {"success": False, "error": str(exc)}

    latency_ms = round((perf_counter() - started) * 1000)
    if result.get("success"):
        connector = await DatabaseFactory.create_connector(**dict(config))
        await DatabaseFactory.mark_connected(database_id, connector)
        profile_result = await DatabaseFactory.refresh_database_profile(database_id)
    else:
        await DatabaseFactory.mark_disconnected(database_id)
        profile_result = None

    latest_config = await DatabaseFactory.get_database(database_id) or config
    return {
        "database": _public_database_config(database_id, latest_config),
        "status": latest_config.get("status", "unknown"),
        "success": bool(result.get("success")),
        "latency_ms": result.get("latency_ms") or latency_ms,
        "version": result.get("version"),
        "error": result.get("error"),
        "profile_refresh": profile_result,
    }


@router.get("/databases/{database_id}/preview")
async def preview_database_resource(database_id: str, refresh: bool = Query(default=False)) -> dict:
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
        schema, profile_cache = await DatabaseFactory.load_schema_with_profile_cache(
            database_id,
            dict(config),
            force_refresh=refresh,
        )
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
            "profile_cache": profile_cache,
        }

    preview = schema_preview(schema)
    return {
        "database": public_config,
        "preview_kind": "schema",
        "summary": f"Loaded {len(preview.get('tables_or_measurements', []))} schema objects.",
        "preview": preview,
        "profile_cache": profile_cache,
    }


@router.get("/fact-memory")
async def get_global_fact_memory() -> dict:
    """Return long-term fact definition and recipe memory."""
    memory = read_fact_memory(None)
    return {
        "memory": memory.model_dump(mode="json"),
        "prompt_view": prompt_fact_memory_view(None),
    }


@router.get("/databases/{database_id}/fact-memory")
async def get_database_fact_memory(database_id: str) -> dict:
    """Return fact definition and recipe memory scoped to one database."""
    config = await DatabaseFactory.get_database(database_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Database '{database_id}' was not found.")
    memory = read_fact_memory(database_id)
    return {
        "database": _public_database_config(database_id, config),
        "memory": memory.model_dump(mode="json"),
        "prompt_view": prompt_fact_memory_view(database_id),
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


async def _list_public_databases() -> list[dict]:
    await DatabaseFactory.list_databases()
    return [
        _public_database_config(db_id, config)
        for db_id, config in sorted(DatabaseFactory._databases.items())
    ]


def _editable_payload_to_config(payload: dict) -> dict:
    extra = payload.pop("extra", None)
    config = {key: value for key, value in payload.items() if value is not None}
    if extra:
        config.update(extra)
    if "type" in config:
        config["type"] = str(config["type"]).strip()
        if not config["type"]:
            raise HTTPException(status_code=422, detail="Database type cannot be empty.")
        try:
            normalized_type = DatabaseFactory.normalize_database_type(config["type"])
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"Unsupported database type: {config['type']}") from exc
        config["type"] = normalized_type
        config["db_type"] = normalized_type
    if "name" in config:
        config["name"] = str(config["name"]).strip()
        if not config["name"]:
            raise HTTPException(status_code=422, detail="Database name cannot be empty.")
    if "display_name" in config:
        config["display_name"] = str(config["display_name"]).strip() or None
    return config

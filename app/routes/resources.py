"""Read-only resource endpoints for the frontend workspace."""
from __future__ import annotations

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
        return {
            "database": public_config,
            "preview_kind": "reference_dataset",
            "summary": "Reference dataset configured for local analysis.",
            "reference_dataset": {
                "dataset_path": reference_dataset.get("dataset_path"),
                "config_source": reference_dataset.get("config_source"),
            },
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

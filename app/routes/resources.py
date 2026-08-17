"""Read-only resource endpoints for the frontend workspace."""
from __future__ import annotations

from time import perf_counter
from typing import Any
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.deps import get_insight_learning_schedule_store
from app.settings import get_settings
from app.model_config import get_model_config_store
from core.database import DatabaseFactory, metric_list_preview, schema_preview
from core.key_insight.memory import (
    database_insight_memory_summary,
    memory_cards_view,
    memory_detail,
    read_persisted_insight_memory,
    system_insight_memory,
)
from core.timeseries.anomaly_registry import (
    available_anomaly_detectors,
    default_anomaly_detector_name,
    register_api_anomaly_detector,
    set_default_anomaly_detector,
    unregister_anomaly_detector,
)
from core.timeseries.forecast_registry import (
    available_forecast_models,
    default_forecast_model_name,
    register_api_forecast_model,
    set_default_forecast_model,
    unregister_forecast_model,
)

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


class InsightLearningScheduleUpdatePayload(BaseModel):
    """Editable automatic Insight learning schedule."""

    max_wait_seconds: float = Field(gt=0, le=7 * 24 * 60 * 60)


class AIModelConfigUpdatePayload(BaseModel):
    """OpenAI-compatible model endpoint configuration."""

    id: str | None = None
    api_base: str = Field(min_length=1)
    model: str = Field(min_length=1)
    api_key: str | None = None


class MachineLearningConfigUpdatePayload(BaseModel):
    """Default registered models used by time-series tools."""

    forecast_model: str = Field(min_length=1)
    anomaly_detector: str = Field(min_length=1)


class ExternalMachineModelPayload(BaseModel):
    """One externally deployed forecast or anomaly model."""

    name: str = Field(min_length=1)
    endpoint: str = Field(min_length=1)
    api_key: str | None = None
    timeout_seconds: float = Field(default=30.0, gt=0, le=300)


class MachineModelConnectionTestPayload(ExternalMachineModelPayload):
    task: str


class ModelConnectionTestPayload(BaseModel):
    """One explicit model connection check requested by the user."""

    kind: str
    connection_id: str | None = None
    api_base: str = Field(min_length=1)
    model: str = Field(min_length=1)
    api_key: str | None = None


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


@router.get("/insight-memory")
async def get_global_insight_memory() -> dict:
    """Return code-owned system Insight Memory only."""
    memory = system_insight_memory()
    view = memory_cards_view(None, max_cards=None)
    return {
        "memory": {
            "cards": view.get("cards", []),
            "storage_path": memory.storage_path,
            "updated_at": memory.updated_at,
        },
        "prompt_view": view,
    }


def _insight_learning_schedule_response() -> dict:
    settings = get_settings()
    schedule = get_insight_learning_schedule_store().read()
    return {
        "settings": {
            "max_wait_seconds": schedule.max_wait_seconds,
            "enabled": settings.insight_memory_learning_enabled,
            "batch_size": settings.insight_memory_learning_batch_size,
        }
    }


@router.get("/insight-memory-learning-settings")
async def get_insight_memory_learning_settings() -> dict:
    """Return the effective automatic Insight learning schedule."""
    return _insight_learning_schedule_response()


@router.patch("/insight-memory-learning-settings")
async def update_insight_memory_learning_settings(payload: InsightLearningScheduleUpdatePayload) -> dict:
    """Persist a schedule update; the running worker observes it on its next poll."""
    get_insight_learning_schedule_store().write(payload)
    return _insight_learning_schedule_response()


@router.get("/insight-memory/{memory_id}")
async def get_global_insight_memory_detail(memory_id: str) -> dict:
    """Return one on-demand insight memory detail."""
    detail = memory_detail(None, memory_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Insight memory '{memory_id}' was not found.")
    return {"detail": detail.model_dump(mode="json")}


@router.get("/databases/{database_id}/insight-memory")
async def get_database_insight_memory(database_id: str) -> dict:
    """Return only Insight Memory learned for one database."""
    config = await DatabaseFactory.get_database(database_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Database '{database_id}' was not found.")
    memory = read_persisted_insight_memory(database_id)
    view = memory_cards_view(database_id, max_cards=None, include_system=False)
    return {
        "database": _public_database_config(database_id, config),
        "memory": {
            "cards": view.get("cards", []),
            "storage_path": memory.storage_path,
            "updated_at": memory.updated_at,
        },
        "prompt_view": view,
    }


@router.get("/databases/{database_id}/insight-memory/{memory_id}")
async def get_database_insight_memory_detail(database_id: str, memory_id: str) -> dict:
    """Return one on-demand insight memory detail scoped to one database."""
    config = await DatabaseFactory.get_database(database_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Database '{database_id}' was not found.")
    detail = memory_detail(database_id, memory_id, include_system=False)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Insight memory '{memory_id}' was not found.")
    return {
        "database": _public_database_config(database_id, config),
        "detail": detail.model_dump(mode="json"),
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
    return {"model": get_model_config_store().effective_ai().model}


@router.get("/models/config")
async def get_models_config() -> dict:
    """Return editable model configuration and registered ML capabilities."""
    config = get_model_config_store().public_config()
    store = get_model_config_store()
    config["machine_learning"] = {
        "forecast_model": default_forecast_model_name(),
        "anomaly_detector": default_anomaly_detector_name(),
        "forecast_options": available_forecast_models(),
        "anomaly_options": available_anomaly_detectors(),
        "forecast_models": store.public_machine_models(
            "forecast", available_forecast_models(), default_forecast_model_name(),
        ),
        "anomaly_models": store.public_machine_models(
            "anomaly", available_anomaly_detectors(), default_anomaly_detector_name(),
        ),
    }
    return config


@router.patch("/models/ai/{section}")
async def update_ai_model_config(section: str, payload: AIModelConfigUpdatePayload) -> dict:
    """Create or update one model connection and refresh dependencies."""
    if section not in {"llm", "embedding"}:
        raise HTTPException(status_code=404, detail=f"Unknown AI model section '{section}'.")
    values = payload.model_dump(exclude_unset=True)
    values["api_base"] = _normalized_api_base(payload.api_base)
    values["model"] = payload.model.strip()
    if payload.api_key is not None:
        values["api_key"] = payload.api_key.strip() or None
    connection_id = get_model_config_store().upsert_ai(section, values)
    from app.deps import clear_runtime_model_dependencies
    clear_runtime_model_dependencies()
    response = await get_models_config()
    response["saved_id"] = connection_id
    return response


@router.patch("/models/ai/{section}/{connection_id}/activate")
async def activate_ai_model_config(section: str, connection_id: str) -> dict:
    """Make one saved connection active for subsequent runtime requests."""
    try:
        get_model_config_store().activate_ai(section, connection_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    from app.deps import clear_runtime_model_dependencies
    clear_runtime_model_dependencies()
    return await get_models_config()


@router.delete("/models/ai/{section}/{connection_id}")
async def delete_ai_model_config(section: str, connection_id: str) -> dict:
    """Remove one workspace-defined model connection."""
    try:
        get_model_config_store().delete_ai(section, connection_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    from app.deps import clear_runtime_model_dependencies
    clear_runtime_model_dependencies()
    return await get_models_config()


@router.patch("/models/machine-learning")
async def update_machine_learning_config(payload: MachineLearningConfigUpdatePayload) -> dict:
    """Select defaults from registered forecast and anomaly implementations."""
    forecast_name = payload.forecast_model.strip().lower()
    anomaly_name = payload.anomaly_detector.strip().lower()
    if forecast_name not in available_forecast_models():
        raise HTTPException(status_code=422, detail=f"Unknown forecast model '{forecast_name}'.")
    if anomaly_name not in available_anomaly_detectors():
        raise HTTPException(status_code=422, detail=f"Unknown anomaly detector '{anomaly_name}'.")
    set_default_forecast_model(forecast_name)
    set_default_anomaly_detector(anomaly_name)
    get_model_config_store().update_machine_learning({
        "forecast_model": default_forecast_model_name(),
        "anomaly_detector": default_anomaly_detector_name(),
    })
    return await get_models_config()


@router.patch("/models/machine-learning/external/{task}")
async def upsert_external_machine_model(task: str, payload: ExternalMachineModelPayload) -> dict:
    """Persist one external model file and register it for immediate tool use."""
    if task not in {"forecast", "anomaly"}:
        raise HTTPException(status_code=404, detail=f"Unknown machine learning task '{task}'.")
    store = get_model_config_store()
    normalized_name = payload.name.strip().lower()
    available = available_forecast_models() if task == "forecast" else available_anomaly_detectors()
    if normalized_name in available and store.external_machine_model(task, normalized_name) is None:
        raise HTTPException(status_code=409, detail=f"Model name '{normalized_name}' is already registered.")
    values = payload.model_dump(exclude_unset=True)
    values["endpoint"] = _normalized_api_base(payload.endpoint)
    if payload.api_key is not None:
        values["api_key"] = payload.api_key.strip() or None
    try:
        config = store.upsert_external_machine_model(task, values)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    headers = {"Authorization": f"Bearer {config['api_key']}"} if config.get("api_key") else {}
    if task == "forecast":
        register_api_forecast_model(config["name"], endpoint=config["endpoint"], timeout_seconds=config["timeout_seconds"], headers=headers)
    else:
        register_api_anomaly_detector(config["name"], endpoint=config["endpoint"], timeout_seconds=config["timeout_seconds"], headers=headers)
    response = await get_models_config()
    response["saved_id"] = config["name"]
    return response


@router.patch("/models/machine-learning/{task}/{name}/activate")
async def activate_machine_model(task: str, name: str) -> dict:
    """Select one registered built-in or external model as the task default."""
    try:
        if task == "forecast":
            set_default_forecast_model(name)
            get_model_config_store().update_machine_learning({"forecast_model": default_forecast_model_name()})
        elif task == "anomaly":
            set_default_anomaly_detector(name)
            get_model_config_store().update_machine_learning({"anomaly_detector": default_anomaly_detector_name()})
        else:
            raise ValueError(f"Unknown machine learning task '{task}'.")
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return await get_models_config()


@router.delete("/models/machine-learning/external/{task}/{name}")
async def delete_external_machine_model(task: str, name: str) -> dict:
    """Unregister and remove one non-active external model config file."""
    active = default_forecast_model_name() if task == "forecast" else default_anomaly_detector_name() if task == "anomaly" else None
    if active is None:
        raise HTTPException(status_code=404, detail=f"Unknown machine learning task '{task}'.")
    if name.strip().lower() == active:
        raise HTTPException(status_code=409, detail="Activate another model before removing this one.")
    try:
        get_model_config_store().delete_external_machine_model(task, name)
        if task == "forecast":
            unregister_forecast_model(name)
        else:
            unregister_anomaly_detector(name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return await get_models_config()


@router.post("/models/machine-learning/test")
async def test_external_machine_model(payload: MachineModelConnectionTestPayload) -> dict:
    """Send a contract-valid sample series to an external ML endpoint."""
    if payload.task not in {"forecast", "anomaly"}:
        raise HTTPException(status_code=422, detail="Task must be 'forecast' or 'anomaly'.")
    existing = get_model_config_store().external_machine_model(payload.task, payload.name)
    api_key = payload.api_key.strip() if payload.api_key else str((existing or {}).get("api_key") or "").strip()
    started = perf_counter()
    try:
        await run_in_threadpool(
            _perform_external_ml_connection_test,
            task=payload.task,
            name=payload.name.strip().lower(),
            endpoint=_normalized_api_base(payload.endpoint),
            api_key=api_key or None,
            timeout_seconds=payload.timeout_seconds,
        )
    except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
        return {"success": False, "latency_ms": round((perf_counter() - started) * 1000, 1), "message": _connection_error_message(exc)}
    return {"success": True, "latency_ms": round((perf_counter() - started) * 1000, 1), "message": f"{payload.name.strip()} returned a valid {payload.task} response."}


@router.post("/models/test")
async def test_model_connection(payload: ModelConnectionTestPayload) -> dict:
    """Run a small, real request against an OpenAI-compatible endpoint."""
    if payload.kind not in {"llm", "embedding"}:
        raise HTTPException(status_code=422, detail="Model kind must be 'llm' or 'embedding'.")
    effective = get_model_config_store().effective_ai()
    saved_key = (
        get_model_config_store().connection_api_key(payload.kind, payload.connection_id)
        if payload.connection_id else None
    )
    api_key = payload.api_key.strip() if payload.api_key else (
        saved_key or (effective.api_key if payload.kind == "llm" else effective.embedding_api_key)
    )
    started = perf_counter()
    try:
        await run_in_threadpool(
            _perform_model_connection_test,
            kind=payload.kind,
            api_base=_normalized_api_base(payload.api_base),
            model=payload.model.strip(),
            api_key=api_key,
        )
    except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
        return {
            "success": False,
            "latency_ms": round((perf_counter() - started) * 1000, 1),
            "message": _connection_error_message(exc),
        }
    return {
        "success": True,
        "latency_ms": round((perf_counter() - started) * 1000, 1),
        "message": f"{payload.model.strip()} responded successfully.",
    }


def _normalized_api_base(value: str) -> str:
    normalized = value.strip().rstrip("/")
    if not normalized.startswith(("http://", "https://")):
        raise HTTPException(status_code=422, detail="API base must use http:// or https://.")
    return normalized


def _perform_model_connection_test(*, kind: str, api_base: str, model: str, api_key: str | None) -> None:
    path = "/chat/completions" if kind == "llm" else "/embeddings"
    body = (
        {"model": model, "messages": [{"role": "user", "content": "Reply with OK."}], "max_tokens": 2}
        if kind == "llm"
        else {"model": model, "input": ["TSPilot connection test"]}
    )
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(
        f"{api_base}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urlopen(request, timeout=20) as response:
        decoded = json.loads(response.read().decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("Model endpoint did not return a JSON object.")


def _perform_external_ml_connection_test(
    *, task: str, name: str, endpoint: str, api_key: str | None, timeout_seconds: float,
) -> None:
    series = {
        "series_name": "connection_test",
        "time_field": "timestamp",
        "value_field": "value",
        "points": [
            {"timestamp": "2026-01-01T00:00:00Z", "value": 1.0},
            {"timestamp": "2026-01-01T01:00:00Z", "value": 2.0},
        ],
        "labels": {"source": "tspilot_connection_test"},
    }
    payload = (
        {"task": "forecast", "model_name": name, "series": series, "horizon": 1, "params": {"connection_test": True}}
        if task == "forecast"
        else {"task": "anomaly", "detector_name": name, "series": series, "params": {"connection_test": True}}
    )
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    with urlopen(request, timeout=timeout_seconds) as response:
        decoded = json.loads(response.read().decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("External model endpoint did not return a JSON object.")
    if task == "forecast" and len(decoded.get("forecast_points") or []) != 1:
        raise ValueError("Forecast endpoint must return one forecast point for the connection test horizon.")
    if task == "anomaly" and not isinstance(decoded.get("anomaly_points", []), list):
        raise ValueError("Anomaly endpoint must return anomaly_points as a list.")


def _connection_error_message(error: Exception) -> str:
    if isinstance(error, HTTPError):
        return f"Endpoint returned HTTP {error.code}."
    return str(error) or error.__class__.__name__


async def _list_public_databases() -> list[dict]:
    await DatabaseFactory.list_databases()
    databases = []
    for db_id, config in sorted(DatabaseFactory._databases.items()):
        database = _public_database_config(db_id, config)
        database["insight_memory_summary"] = database_insight_memory_summary(db_id)
        databases.append(database)
    return databases


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

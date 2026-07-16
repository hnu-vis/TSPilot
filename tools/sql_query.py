"""Unified database query tool."""
from __future__ import annotations

import re

from pydantic import BaseModel, Field, model_validator

from app.settings import Settings
from core.database import DatabaseFactory, execute_query, normalize_query_result
from core.database.repair import classify_query_error, repair_read_only_query
from schemas.database_context import DatabaseContext
from tools.base import BaseTool


class _ExplicitQueryInput(BaseModel):
    database_context: DatabaseContext
    query: str
    query_language: str | None = None
    purpose: str | None = None
    constraints: dict = Field(default_factory=dict)


class SqlQueryInput(BaseModel):
    message: str | None = None
    database_context: DatabaseContext
    time_range: dict | None = None
    constraints: dict = Field(default_factory=dict)
    intent_profile: dict = Field(default_factory=dict)
    selected_database: str | None = None
    selected_database_type: str | None = None
    history: list[dict] = Field(default_factory=list)
    query: str | None = None
    query_language: str | None = None
    purpose: str | None = None

    @model_validator(mode="after")
    def require_message_or_query(self):
        if not (self.query and self.query.strip()) and not (self.message and self.message.strip()):
            raise ValueError("sql_query requires either message for automatic planning or query for explicit read-only execution.")
        return self


class _ExplicitQueryExecutor(BaseTool):
    """Run a safe model-authored read-only query."""

    _SQL_WRITE_PATTERN = re.compile(
        r"\b(insert|update|delete|drop|alter|truncate|create|replace|merge|grant|revoke|vacuum|attach|detach|pragma)\b",
        re.IGNORECASE,
    )
    _FLUX_WRITE_PATTERN = re.compile(r"\bto\s*\(|experimental\.to\s*\(", re.IGNORECASE)

    def __init__(self, settings: Settings):
        self._settings = settings

    async def execute(self, validated_input: _ExplicitQueryInput, **kwargs) -> dict:
        repair = repair_read_only_query(
            query=validated_input.query,
            query_language=validated_input.query_language,
        )
        query = repair.query
        self._validate_read_only(query, validated_input.query_language)
        config = await self._load_database_config(validated_input.database_context.database_id)
        connector = await DatabaseFactory.create_connector(**config)
        async with connector:
            result, executed_query, repair_diagnostics = await self._execute_with_repair(
                connector=connector,
                query=query,
                query_language=validated_input.query_language,
                timeout=int(validated_input.constraints.get("timeout", config.get("query_timeout", 60))),
                initial_repair_reason=repair.reason,
            )
        evidence = normalize_query_result(
            database_id=validated_input.database_context.database_id,
            database_type=str(config.get("type", validated_input.database_context.database_type)),
            query_language=validated_input.query_language or self._infer_query_language(config),
            query=executed_query,
            result=result,
        )
        evidence.metadata = {
            **evidence.metadata,
            "sql_query_mode": "explicit",
            "purpose": validated_input.purpose,
        }
        evidence.diagnostics = {
            **evidence.diagnostics,
            "sql_query": {
                "purpose": validated_input.purpose,
                "row_count": result.row_count,
                "columns": result.columns,
                "truncated": result.truncated,
                "execution_time_ms": result.execution_time_ms,
                "repair": repair_diagnostics,
            },
        }
        return evidence.model_dump(mode="json")

    async def _execute_with_repair(
        self,
        *,
        connector,
        query: str,
        query_language: str | None,
        timeout: int,
        initial_repair_reason: str | None,
    ):
        repair_diagnostics = {
            "attempted": bool(initial_repair_reason),
            "initial_repair": initial_repair_reason,
            "retry_repair": None,
            "retry_hint": None,
            "error_classification": None,
        }
        try:
            return await execute_query(connector, query, timeout=timeout), query, repair_diagnostics
        except Exception as exc:
            classification = classify_query_error(exc)
            retry_repair = repair_read_only_query(
                query=query,
                query_language=query_language,
                error=exc,
            )
            repair_diagnostics["attempted"] = repair_diagnostics["attempted"] or retry_repair.changed
            repair_diagnostics["retry_repair"] = retry_repair.reason
            repair_diagnostics["retry_hint"] = retry_repair.hint or classification.get("suggestion")
            repair_diagnostics["error_classification"] = classification
            if not retry_repair.changed or retry_repair.query == query:
                raise RuntimeError(
                    f"{classification['message']} "
                    f"Repair hint: {repair_diagnostics['retry_hint'] or 'rewrite the read-only query based on the error.'}"
                ) from exc
            self._validate_read_only(retry_repair.query, query_language)
            try:
                return await execute_query(connector, retry_repair.query, timeout=timeout), retry_repair.query, repair_diagnostics
            except Exception as retry_exc:
                retry_classification = classify_query_error(retry_exc)
                repair_diagnostics["error_classification"] = retry_classification
                raise RuntimeError(
                    f"{retry_classification['message']} "
                    f"Repair hint: {retry_repair.hint or retry_classification.get('suggestion') or 'rewrite the read-only query based on the error.'}"
                ) from retry_exc

    async def _load_database_config(self, database_id: str) -> dict:
        await DatabaseFactory.load_databases()
        config = await DatabaseFactory.get_database(database_id)
        if not config:
            raise FileNotFoundError(
                f"Database config for '{database_id}' was not found in {self._settings.resolved_database_config_dir}"
            )
        return dict(config)

    def _validate_read_only(self, query: str, query_language: str | None) -> None:
        stripped = query.strip()
        if not stripped:
            raise ValueError("sql_query requires a non-empty query.")
        normalized_language = str(query_language or "").lower()
        if normalized_language in {"sql", "sqlite", "postgresql", "timescaledb", "questdb", "clickhouse"}:
            if not re.match(r"^\s*(with|select)\b", stripped, re.IGNORECASE):
                raise ValueError("Only read-only SELECT/ WITH SQL analysis queries are allowed.")
            if self._SQL_WRITE_PATTERN.search(stripped):
                raise ValueError("Write or DDL statements are not allowed in sql_query.")
            return
        if normalized_language == "flux" or "|>" in stripped or stripped.startswith("from("):
            if self._FLUX_WRITE_PATTERN.search(stripped):
                raise ValueError("Flux output/write functions are not allowed in sql_query.")
            return
        if self._SQL_WRITE_PATTERN.search(stripped):
            raise ValueError("Write or DDL statements are not allowed in sql_query.")

    def _infer_query_language(self, config: dict) -> str:
        db_type = str(config.get("type") or config.get("db_type") or "")
        if db_type == "influxdb":
            return "flux"
        if db_type == "prometheus":
            return "promql"
        return "sql"


class SqlQueryTool(BaseTool):
    """Unified database query tool for planned and explicit read-only queries."""

    def __init__(self, settings: Settings):
        from tools.query_database import QueryDatabaseTool

        self._planned_query_tool = QueryDatabaseTool(settings)
        self._explicit_query_executor = _ExplicitQueryExecutor(settings)

    async def execute(self, validated_input: SqlQueryInput, **kwargs) -> dict:
        request_state = kwargs.get("request_state")
        intent_profile = validated_input.intent_profile or getattr(request_state, "intent_profile", {}) or {}
        if validated_input.query and validated_input.query.strip():
            return await self._explicit_query_executor.execute(
                _ExplicitQueryInput(
                    database_context=validated_input.database_context,
                    query=validated_input.query,
                    query_language=validated_input.query_language,
                    purpose=validated_input.purpose,
                    constraints=validated_input.constraints,
                ),
                **kwargs,
            )

        from tools.query_database import QueryDatabaseInput

        return await self._planned_query_tool.execute(
            QueryDatabaseInput(
                message=validated_input.message or "",
                database_context=validated_input.database_context,
                time_range=validated_input.time_range,
                constraints=validated_input.constraints,
                intent_profile=intent_profile,
                selected_database=validated_input.selected_database,
                selected_database_type=validated_input.selected_database_type,
                history=validated_input.history,
            ),
            **kwargs,
        )

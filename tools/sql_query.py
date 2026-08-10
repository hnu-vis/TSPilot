"""Unified database query tool."""
from __future__ import annotations

import re
import time
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, Field, field_validator, model_validator

from app.settings import Settings
from core.database import DatabaseFactory, execute_query, normalize_query_result
from core.database.dialects import dialect_for_database, query_language_for_database_type
from core.database.llm_query import LLMGeneratedQuery, LLMQueryGenerationResult, LLMQueryGenerator
from core.database.query_errors import classify_query_error
from core.database.schema import schema_preview
from schemas.state import RequestStateModel
from schemas.database_context import DatabaseContext
from schemas.data_fact import DataFactRequest
from core.data_fact.contracts import fact_request_contract_error
from tools.base import BaseTool, StructuredToolError


class _ExplicitQueryInput(BaseModel):
    database_context: DatabaseContext
    query: str
    query_language: str | None = None
    purpose: str | None = None
    constraints: dict = Field(default_factory=dict)
    fact_requests: list[DataFactRequest] = Field(default_factory=list)


class SqlQueryInput(BaseModel):
    mode: str | None = None
    repair_contract: dict | None = None
    message: str | None = None
    database_context: DatabaseContext
    time_range: dict | None = None
    constraints: dict = Field(default_factory=dict)
    intent_profile: dict = Field(default_factory=dict)
    selected_database: str | None = None
    selected_database_type: str | None = None
    history: list[dict] = Field(default_factory=list)
    fact_requests: list[DataFactRequest] = Field(default_factory=list)
    query: str | None = None
    query_language: str | None = None
    purpose: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_query_message_union_alias(cls, data):
        if not isinstance(data, dict):
            return data
        if "message|query" not in data:
            return data
        normalized = dict(data)
        value = normalized.pop("message|query")
        text = str(value or "").strip()
        if not text:
            return normalized
        lowered = text.lower()
        if "|>" in text or lowered.startswith(("from(", "select ", "with ")):
            normalized.setdefault("query", text)
        else:
            normalized.setdefault("message", text)
        return normalized

    @field_validator("fact_requests", mode="before")
    @classmethod
    def keep_only_valid_fact_request_hints(cls, value):
        if value in (None, ""):
            return []
        if not isinstance(value, list):
            return []
        normalized: list = []
        for item in value:
            if isinstance(item, DataFactRequest):
                normalized.append(item)
                continue
            if not isinstance(item, dict):
                continue
            if item.get("name") and item.get("fact_type"):
                normalized.append(item)
        return normalized

    @model_validator(mode="after")
    def require_message_or_query(self):
        supported_fact_requests: list[DataFactRequest] = []
        unsupported_fact_requests: list[dict] = []
        for request in self.fact_requests:
            error = fact_request_contract_error(request, "sql_query")
            if error:
                unsupported_fact_requests.append(
                    {**request.model_dump(mode="json", exclude_none=True), "contract_error": error}
                )
            else:
                supported_fact_requests.append(request)
        self.fact_requests = supported_fact_requests
        if unsupported_fact_requests:
            self.constraints = {
                **self.constraints,
                "unsupported_fact_requests": unsupported_fact_requests,
            }
        has_repair_contract = str(self.mode or "").strip().lower() == "repair" and isinstance(self.repair_contract, dict)
        if not has_repair_contract and not (self.query and self.query.strip()) and not (self.message and self.message.strip()):
            raise ValueError("sql_query requires either message for automatic planning or query for explicit read-only execution.")
        return self


class _ExplicitQueryExecutor(BaseTool):
    """Run a safe model-authored read-only query."""

    def __init__(self, settings: Settings):
        self._settings = settings

    async def execute(self, validated_input: _ExplicitQueryInput, **kwargs) -> dict:
        return await self.execute_query_input(validated_input, mode="explicit")

    async def execute_query_input(
        self,
        validated_input: _ExplicitQueryInput,
        *,
        mode: str,
        extra_metadata: dict | None = None,
        extra_diagnostics: dict | None = None,
        **kwargs,
    ) -> dict:
        query = validated_input.query
        try:
            self._validate_read_only(query, validated_input.query_language)
        except ValueError as exc:
            query_language = validated_input.query_language
            repair_contract = {
                "mode": "query_repair",
                "previous_query": query,
                "query_language": query_language,
                "required_contract": {"read_only": True},
            }
            validation_failure = {
                "scope": "query_validation",
                "capability": "query",
                "tool": "sql_query",
                "error_code": "query_read_only_violation",
                "message": str(exc),
                "failed_artifact": {"query": query, "query_language": query_language},
                "required_contract": {"read_only": True},
                "repair_contract": repair_contract,
                "retry_policy": {
                    "required_action": "sql_query",
                    "max_equivalent_retries": 1,
                    "allow_same_action": True,
                    "terminal_after_exhausted": True,
                },
            }
            raise StructuredToolError(
                f"Query validation failed: {exc}",
                error_type="query_read_only_violation",
                retryable=True,
                recommended_next_action="sql_query",
                diagnostics={"query": query, "query_language": query_language, "repair_contract": repair_contract},
                validation_failure=validation_failure,
            ) from exc
        config = await self._load_database_config(validated_input.database_context.database_id)
        connector = await DatabaseFactory.create_connector(**config)
        async with connector:
            request_state = kwargs.get("request_state")
            await self._validate_required_filters(
                connector=connector,
                config=config,
                validated_input=validated_input,
                query=query,
                request_state=request_state if isinstance(request_state, RequestStateModel) else None,
            )
            try:
                result = await self._execute_connector_query(
                    connector=connector,
                    query=query,
                    query_language=validated_input.query_language or self._infer_query_language(config),
                    constraints=validated_input.constraints,
                    timeout=int(validated_input.constraints.get("timeout", config.get("query_timeout", 60))),
                )
            except StructuredToolError:
                raise
            except Exception as exc:
                query_language = validated_input.query_language or self._infer_query_language(config)
                classification = classify_query_error(exc)
                repair_contract = {
                    "mode": "query_repair",
                    "previous_query": query,
                    "query_language": query_language,
                    "execution_error": str(exc),
                    "schema_linking_contract": validated_input.constraints.get("_schema_linking_contract")
                    if isinstance(validated_input.constraints, dict)
                    else {},
                }
                validation_failure = {
                    "scope": "query_execution",
                    "capability": "query",
                    "tool": "sql_query",
                    "error_code": "query_execution_failed",
                    "message": str(exc),
                    "failed_artifact": {"query": query, "query_language": query_language},
                    "required_contract": {"read_only": True},
                    "repair_contract": repair_contract,
                    "retry_policy": {
                        "required_action": "sql_query",
                        "max_equivalent_retries": 2,
                        "allow_same_action": True,
                        "terminal_after_exhausted": True,
                    },
                }
                raise StructuredToolError(
                    f"Explicit query execution failed: {exc}",
                    error_type="query_execution_failed",
                    retryable=bool(classification.get("retryable", True)),
                    recommended_next_action="sql_query",
                    diagnostics={
                        "query_language": query_language,
                        "query": query,
                        "classification": classification,
                        "repair_contract": repair_contract,
                    },
                    validation_failure=validation_failure,
                ) from exc
        evidence = normalize_query_result(
            database_id=validated_input.database_context.database_id,
            database_type=str(config.get("type", validated_input.database_context.database_type)),
            query_language=validated_input.query_language or self._infer_query_language(config),
            query=query,
            result=result,
        )
        evidence.metadata = {
            **evidence.metadata,
            "sql_query_mode": mode,
            "purpose": validated_input.purpose,
            **(extra_metadata or {}),
        }
        evidence.diagnostics = {
            **evidence.diagnostics,
            "sql_query": {
                "purpose": validated_input.purpose,
                "row_count": result.row_count,
                "columns": result.columns,
                "truncated": result.truncated,
                "execution_time_ms": result.execution_time_ms,
                "repair": {"attempted": False, "strategy": "react"},
            },
            **(extra_diagnostics or {}),
        }
        evidence.diagnostics["task_coverage"] = self._task_coverage_diagnostics(
            validated_input=validated_input,
            evidence=evidence.model_dump(mode="json"),
            query=query,
            query_language=validated_input.query_language or self._infer_query_language(config),
            base=(extra_diagnostics or {}).get("task_coverage") if isinstance(extra_diagnostics, dict) else None,
            selected_fields=(
                ((extra_diagnostics or {}).get("llm_query_generation") or {}).get("selected_fields")
                if isinstance(extra_diagnostics, dict)
                and isinstance((extra_diagnostics or {}).get("llm_query_generation"), dict)
                else None
            ),
        )
        return evidence.model_dump(mode="json")

    async def _execute_connector_query(
        self,
        *,
        connector,
        query: str,
        query_language: str,
        constraints: dict,
        timeout: int,
    ):
        execution = constraints.get("_query_execution") if isinstance(constraints, dict) else None
        if (
            query_language == "promql"
            and isinstance(execution, dict)
            and str(execution.get("mode") or "instant").lower() == "range"
            and callable(getattr(connector, "get_range", None))
        ):
            end = self._execution_time(execution.get("end")) or datetime.now(timezone.utc)
            start = self._execution_time(execution.get("start"))
            if start is None:
                start = end - timedelta(seconds=int(execution["lookback_seconds"]))
            step_seconds = int(execution.get("step_seconds") or 15)
            return await connector.get_range(query, start=start, end=end, step=f"{step_seconds}s")
        return await execute_query(connector, query, timeout=timeout)

    def _execution_time(self, value) -> datetime | None:
        if value in (None, ""):
            return None
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    async def _load_database_config(self, database_id: str) -> dict:
        await DatabaseFactory.load_databases()
        config = await DatabaseFactory.get_database(database_id)
        if not config:
            raise FileNotFoundError(
                f"Database config for '{database_id}' was not found in {self._settings.resolved_database_config_dir}"
            )
        return dict(config)

    def _validate_read_only(self, query: str, query_language: str | None) -> None:
        dialect_for_database(self._database_type_from_language(query_language, query)).validate_read_only(query, query_language)

    def _infer_query_language(self, config: dict) -> str:
        db_type = str(config.get("type") or config.get("db_type") or "")
        return query_language_for_database_type(db_type)

    def _database_type_from_language(self, query_language: str | None, query: str | None = None) -> str:
        language = str(query_language or "").strip().lower()
        query_text = str(query or "").strip()
        if language == "flux" or "|>" in query_text or query_text.startswith("from("):
            return "influxdb"
        if language == "promql":
            return "prometheus"
        return language or "sql"

    async def _validate_required_filters(
        self,
        *,
        connector,
        config: dict,
        validated_input: _ExplicitQueryInput,
        query: str,
        request_state: RequestStateModel | None,
    ) -> None:
        contract_filters = self._contract_required_filters(validated_input.constraints)
        if not contract_filters:
            return
        query_language = validated_input.query_language or self._infer_query_language(config)
        missing = self._missing_rendered_required_filters(query, query_language, contract_filters)
        if missing:
            self._raise_missing_required_filters(
                missing,
                schema_linking_contract=validated_input.constraints.get("_schema_linking_contract"),
                query=query,
                query_language=query_language,
            )

    def _contract_required_filters(self, constraints: dict) -> list[dict]:
        contract = constraints.get("_schema_linking_contract") if isinstance(constraints, dict) else None
        if not isinstance(contract, dict):
            return []
        raw_filters = contract.get("required_filters")
        if not isinstance(raw_filters, list):
            return []
        filters: list[dict] = []
        for item in raw_filters:
            if not isinstance(item, dict):
                continue
            column = str(item.get("column") or "").strip()
            if not column:
                continue
            filters.append({
                "source": item.get("source"),
                "column": column,
                "operator": str(item.get("operator") or "="),
                "value": item.get("value"),
            })
        return filters

    def _missing_rendered_required_filters(self, query: str, query_language: str, required_filters: list[dict]) -> list[dict]:
        dialect = dialect_for_database(query_language)
        return [
            item for item in required_filters
            if not dialect.has_filter(query, column=item["column"], value=item.get("value"))
        ]

    def _raise_missing_required_filters(
        self,
        missing: list[dict],
        *,
        schema_linking_contract: dict | None,
        details: str | None = None,
        query: str | None = None,
        query_language: str | None = None,
    ) -> None:
        rendered_details = details or "; ".join(
            f"Rendered query is missing the required filter {item['column']}={item.get('value')!r}."
            for item in missing
        )
        missing_filters = [dict(item) for item in missing]
        repair_contract = {
            "mode": "query_repair",
            "previous_query": query,
            "query_language": query_language,
            "must_preserve_filters": missing_filters,
            "schema_linking_contract": schema_linking_contract or {},
            "forbidden_failure_signature": "required_filter_missing:" + ",".join(
                f"{item['column']}={item['value']}" for item in missing_filters
            ),
        }
        validation_failure = {
            "scope": "query_validation",
            "capability": "query",
            "tool": "sql_query",
            "error_code": "required_filter_missing",
            "message": rendered_details,
            "failed_artifact": {"query": query, "query_language": query_language},
            "required_contract": {"required_filters": missing_filters},
            "repair_contract": repair_contract,
            "retry_policy": {
                "required_action": "sql_query",
                "max_equivalent_retries": 2,
                "allow_same_action": True,
                "terminal_after_exhausted": True,
            },
        }
        raise StructuredToolError(
            "Explicit query is missing filters required by the user request. "
            f"{rendered_details} Preserve those filters or use sql_query automatic planning.",
            error_type="required_filter_missing",
            retryable=True,
            recommended_next_action="sql_query",
            diagnostics={
                "missing_required_filters": missing_filters,
                "schema_linking": schema_linking_contract or {},
                "repair_contract": repair_contract,
            },
            validation_failure=validation_failure,
        )

    def _task_coverage_diagnostics(
        self,
        *,
        validated_input: _ExplicitQueryInput,
        evidence: dict,
        query: str,
        query_language: str,
        base: dict | None,
        selected_fields: list[str] | None = None,
    ) -> dict:
        data = evidence.get("data") if isinstance(evidence.get("data"), dict) else {}
        diagnostics = evidence.get("diagnostics") if isinstance(evidence.get("diagnostics"), dict) else {}
        rows = data.get("rows") if isinstance(data.get("rows"), list) else []
        points = data.get("points") if isinstance(data.get("points"), list) else []
        row_count = diagnostics.get("row_count_total")
        if row_count is None:
            row_count = len(rows)
        result_summary = {
            "result_type": evidence.get("result_type"),
            "columns": (evidence.get("columns") or [])[:40],
            "row_count": row_count,
            "visible_row_count": len(rows),
            "point_count": len(points),
            "is_full_fidelity": diagnostics.get("is_full_fidelity"),
            "truncated": diagnostics.get("truncated"),
        }
        satisfied = self._string_list((base or {}).get("satisfied"))
        missing = self._coverage_missing_items(base)
        runtime_missing = self._runtime_missing_items(
            selected_fields=selected_fields,
            columns=evidence.get("columns") or [],
            row_count=row_count,
            query=query,
            query_language=query_language,
        )
        if self._raw_limit_timeseries_risk(
            purpose=validated_input.purpose,
            evidence=evidence,
            query=query,
            query_language=query_language,
        ):
            runtime_missing.append(
                "time-series evidence uses raw LIMIT; use full-range aggregation or representative downsampling instead"
            )
        for item in runtime_missing:
            if item not in missing:
                missing.append(item)
        if row_count == 0 and "query returned no rows" not in missing:
            missing.append("query returned no rows")
            runtime_missing.append("query returned no rows")
        if not satisfied and validated_input.purpose:
            satisfied.append(f"executed query for: {validated_input.purpose}")
        next_action_hint = self._optional_string((base or {}).get("next_action_hint"))
        if missing and not next_action_hint:
            next_action_hint = "Use the latest query, schema linking, and result summary to issue another focused sql_query for the missing facts."
        return {
            "source": (base or {}).get("source") or "sql_query_runtime",
            "user_request": validated_input.purpose,
            "executed_query": query,
            "query_language": query_language,
            "result_summary": result_summary,
            "satisfied": satisfied,
            "missing": missing,
            "runtime_missing": runtime_missing,
            "next_action_hint": next_action_hint,
            "requires_followup": bool(missing),
            "runtime_requires_followup": bool(runtime_missing),
        }

    def _runtime_missing_items(
        self,
        *,
        selected_fields: list[str] | None,
        columns: list,
        row_count: int,
        query: str | None = None,
        query_language: str | None = None,
    ) -> list[str]:
        if row_count == 0:
            return []
        selected_expected = {
            str(field).strip().lower()
            for field in (selected_fields or [])
            if str(field).strip()
        }
        projected = self._projected_columns_from_query(query=query, query_language=query_language)
        expected = projected or selected_expected
        if not expected:
            return []
        actual = {
            str(column).strip().lower()
            for column in columns
            if str(column).strip()
        }
        dialect = dialect_for_database(query_language)
        if dialect.has_value_alias(actual):
            return []
        missing_fields = sorted(
            field
            for field in expected
            if not dialect.field_present_in_columns(field, actual)
        )
        if not missing_fields:
            return []
        return [
            "selected result fields are not present in returned columns: "
            + ", ".join(missing_fields[:8])
        ]

    def _coverage_missing_items(self, coverage: dict | None) -> list[str]:
        if not isinstance(coverage, dict):
            return []
        missing = self._string_list(coverage.get("missing"))
        if missing:
            return missing
        return self._string_list(coverage.get("missing_or_uncertain"))

    def _raw_limit_timeseries_risk(
        self,
        *,
        purpose: str | None,
        evidence: dict,
        query: str | None,
        query_language: str | None,
    ) -> bool:
        if evidence.get("result_type") != "timeseries":
            return False
        if not dialect_for_database(query_language).raw_limit_without_downsampling(query, query_language):
            return False
        normalized_purpose = str(purpose or "").lower()
        return any(
            token in normalized_purpose
            for token in (
                "forecast",
                "predict",
                "prediction",
                "trend",
                "anomaly",
                "outlier",
                "seasonality",
                "预测",
                "趋势",
                "异常",
                "离群",
                "周期",
            )
        )

    def _projected_columns_from_query(self, *, query: str | None, query_language: str | None) -> set[str]:
        return dialect_for_database(query_language).projected_columns(query=query, query_language=query_language)

    def _string_list(self, value) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    def _optional_string(self, value) -> str | None:
        if value in (None, ""):
            return None
        text = str(value).strip()
        return text or None


class SqlQueryTool(BaseTool):
    """Unified database query tool for planned and explicit read-only queries."""

    def __init__(self, settings: Settings, llm=None):
        self._explicit_query_executor = _ExplicitQueryExecutor(settings)
        self._llm_query_generator = LLMQueryGenerator(llm) if llm is not None else None
        self._settings = settings

    async def execute(self, validated_input: SqlQueryInput, **kwargs) -> dict:
        if validated_input.query and validated_input.query.strip() and str(validated_input.mode or "").strip().lower() != "repair":
            return await self._explicit_query_executor.execute_query_input(
                _ExplicitQueryInput(
                    database_context=validated_input.database_context,
                    query=validated_input.query,
                    query_language=validated_input.query_language,
                    purpose=validated_input.purpose,
                    constraints=validated_input.constraints,
                    fact_requests=validated_input.fact_requests,
                ),
                mode="explicit",
                **kwargs,
            )

        return await self._execute_llm_planned_query(validated_input, **kwargs)

    async def _execute_llm_planned_query(self, validated_input: SqlQueryInput, **kwargs) -> dict:
        if self._llm_query_generator is None:
            raise RuntimeError("sql_query automatic mode requires an LLM query generator.")

        total_started = time.perf_counter()
        config = await self._explicit_query_executor._load_database_config(
            validated_input.database_context.database_id
        )
        schema, preview = await self._load_schema_and_preview(validated_input, config)
        request_state = kwargs.get("request_state")
        grounding_message = self._grounding_message(
            validated_input,
            request_state if isinstance(request_state, RequestStateModel) else None,
        )
        llm_linking_started = time.perf_counter()
        schema_linking_contract = None
        try:
            schema_linking_contract = await self._llm_query_generator.generate_schema_linking(
                database_id=validated_input.database_context.database_id,
                database_type=str(config.get("type", validated_input.database_context.database_type)),
                message=grounding_message,
                schema_preview=preview,
                time_range=validated_input.time_range,
                constraints=validated_input.constraints,
                history=validated_input.history,
                request_state=request_state,
            )
        except Exception as exc:
            raise StructuredToolError(
                f"sql_query schema linking failed: {exc}",
                error_type="schema_linking_failed",
                retryable=True,
                recommended_next_action="sql_query",
                diagnostics={
                    "grounding_message": grounding_message,
                },
            ) from exc
        llm_linking_duration_ms = int((time.perf_counter() - llm_linking_started) * 1000)
        preview = {
            **preview,
            "schema_linking": schema_linking_contract,
        }
        generation_started = time.perf_counter()
        repair_contract = (
            validated_input.repair_contract
            if isinstance(validated_input.repair_contract, dict)
            else validated_input.constraints.get("_repair_contract")
            if isinstance(validated_input.constraints, dict) and isinstance(validated_input.constraints.get("_repair_contract"), dict)
            else None
        )
        repair_failure = (
            validated_input.constraints.get("_validation_failure")
            if isinstance(validated_input.constraints, dict) and isinstance(validated_input.constraints.get("_validation_failure"), dict)
            else None
        )
        try:
            generation = await self._llm_query_generator.generate(
                database_id=validated_input.database_context.database_id,
                database_type=str(config.get("type", validated_input.database_context.database_type)),
                message=grounding_message,
                schema_preview=preview,
                time_range=validated_input.time_range,
                constraints=validated_input.constraints,
                history=validated_input.history,
                fact_requests=[item.model_dump(mode="json", exclude_none=True) for item in validated_input.fact_requests],
                previous_query=repair_contract.get("previous_query") if repair_contract else None,
                error=repair_failure or {"repair_contract": repair_contract} if repair_contract else None,
                request_state=request_state,
            )
        except Exception as exc:
            raise StructuredToolError(
                f"sql_query query generation failed: {exc}",
                error_type="query_generation_failed",
                retryable=True,
                recommended_next_action="sql_query",
                diagnostics={
                    "schema_linking": schema_linking_contract,
                    "grounding_message": grounding_message,
                    "runtime_ms": {
                    "schema_linking_llm_duration_ms": llm_linking_duration_ms,
                        "query_generation_duration_ms": int((time.perf_counter() - generation_started) * 1000),
                    },
                },
            ) from exc
        generation_duration_ms = int((time.perf_counter() - generation_started) * 1000)
        runtime_diagnostics = {
            "schema_linking_llm_duration_ms": llm_linking_duration_ms,
            "query_generation_duration_ms": generation_duration_ms,
            "sql_query_planning_duration_ms": int((time.perf_counter() - total_started) * 1000),
        }
        return await self._execute_generated_query(
            validated_input,
            config,
            generation,
            schema_linking_diagnostics=schema_linking_contract,
            extra_runtime_diagnostics=runtime_diagnostics,
            **kwargs,
        )

    async def _load_schema_and_preview(self, validated_input: SqlQueryInput, config: dict):
        try:
            schema, _profile_cache = await DatabaseFactory.load_schema_with_profile_cache(
                validated_input.database_context.database_id,
                dict(config),
            )
            dialect = dialect_for_database(str(config.get("type", validated_input.database_context.database_type)))
            return schema, schema_preview(schema, dialect=dialect)
        except Exception:
            if validated_input.database_context.schema_hint:
                return None, validated_input.database_context.schema_hint
            raise

    async def _load_schema_preview(self, validated_input: SqlQueryInput, config: dict) -> dict:
        _, preview = await self._load_schema_and_preview(validated_input, config)
        return preview

    def _grounding_message(
        self,
        validated_input: SqlQueryInput,
        request_state: RequestStateModel | None,
    ) -> str:
        user_message = (request_state.message if request_state is not None else "") or ""
        tool_message = validated_input.message or ""
        if user_message.strip() and tool_message.strip() and user_message.strip() != tool_message.strip():
            return f"User request: {user_message.strip()}\nTool query purpose: {tool_message.strip()}"
        return user_message.strip() or tool_message.strip()

    async def _execute_generated_query(
        self,
        validated_input: SqlQueryInput,
        config: dict,
        generation: LLMQueryGenerationResult,
        *,
        previous_error: Exception | None = None,
        schema_linking_diagnostics: dict | None = None,
        extra_runtime_diagnostics: dict | None = None,
        **kwargs,
    ) -> dict:
        generated = generation.generated_query
        self._validate_generated_query_shape(
            generated,
            config,
            constraints=validated_input.constraints,
            schema_linking_diagnostics=schema_linking_diagnostics,
        )
        query_language = generated.query_language or self._infer_query_language(config)
        contract = (
            generated.query_task_contract.model_dump(mode="json")
            if hasattr(generated.query_task_contract, "model_dump")
            else generated.query_task_contract
        )
        try:
            return await self._explicit_query_executor.execute_query_input(
                _ExplicitQueryInput(
                    database_context=validated_input.database_context,
                    query=generated.query,
                    query_language=query_language,
                    purpose=generated.purpose,
                    constraints={
                        **validated_input.constraints,
                        **({"_schema_linking_contract": schema_linking_diagnostics} if schema_linking_diagnostics else {}),
                        "_query_execution": generated.query_execution.model_dump(mode="json", exclude_none=True),
                    },
                    fact_requests=validated_input.fact_requests,
                ),
                mode="llm",
                extra_metadata={
                    "generation_mode": "llm",
                    "expected_result_type": generated.expected_result_type,
                    "query_execution": generated.query_execution.model_dump(mode="json", exclude_none=True),
                },
                extra_diagnostics={
                    **self._generation_diagnostics(generation, previous_error=previous_error),
                    **({"schema_linking_generation": schema_linking_diagnostics} if schema_linking_diagnostics else {}),
                    **({"runtime_ms": extra_runtime_diagnostics} if extra_runtime_diagnostics else {}),
                },
                **kwargs,
            )
        except StructuredToolError:
            raise
        except Exception as exc:
            classification = classify_query_error(exc)
            recommended = contract.get("downstream_action") if isinstance(contract, dict) else None
            raise StructuredToolError(
                f"Generated query execution failed: {exc}",
                error_type="query_execution_failed",
                retryable=bool(classification.get("retryable", True)),
                recommended_next_action="sql_query",
                diagnostics={
                    "query_language": query_language,
                    "query": generated.query,
                    "query_task_contract": contract if isinstance(contract, dict) else None,
                    "classification": classification,
                    "recommended_downstream_action": recommended,
                    "strategy_hint": (
                        "If this query tried to compute derived or multi-metric outputs in the database, "
                        "ask sql_query for simpler raw evidence and use the downstream tool named by query_task_contract."
                    ),
                },
            ) from exc

    def _validate_generated_query_shape(
        self,
        generated: LLMGeneratedQuery,
        config: dict,
        *,
        constraints: dict | None = None,
        schema_linking_diagnostics: dict | None = None,
    ) -> None:
        query_language = generated.query_language or self._infer_query_language(config)
        contract = (
            generated.query_task_contract.model_dump(mode="json")
            if hasattr(generated.query_task_contract, "model_dump")
            else generated.query_task_contract
        )
        if isinstance(constraints, dict) and constraints.get("evidence_shape") == "raw_timeseries":
            contract = dict(contract or {})
            contract["preferred_evidence_shape"] = "raw_series"
            contract.setdefault("downstream_action", "code_interpreter")
        if schema_linking_diagnostics:
            contract = dict(contract or {})
            contract["_schema_linking_diagnostics"] = schema_linking_diagnostics
        dialect = dialect_for_database(query_language)
        issues = dialect.query_shape_issues(
            query=generated.query,
            query_language=query_language,
            query_task_contract=contract if isinstance(contract, dict) else None,
        )
        if not issues:
            return
        recommended = None
        if isinstance(contract, dict):
            recommended = contract.get("downstream_action") or None
        repair_contract = {
            "mode": "query_repair",
            "previous_query": generated.query,
            "query_language": query_language,
            "query_shape_issues": issues,
            "query_task_contract": contract if isinstance(contract, dict) else {},
        }
        validation_failure = {
            "scope": "query_generation",
            "capability": "query",
            "tool": "sql_query",
            "error_code": "query_shape_invalid",
            "message": "Generated query does not satisfy the dialect/query-task contract.",
            "failed_artifact": {"query": generated.query, "query_language": query_language},
            "required_contract": contract if isinstance(contract, dict) else {},
            "repair_contract": repair_contract,
            "retry_policy": {
                "required_action": "sql_query",
                "max_equivalent_retries": 2,
                "allow_same_action": True,
                "terminal_after_exhausted": True,
            },
        }
        raise StructuredToolError(
            "Generated query does not satisfy the dialect/query-task contract.",
            error_type="query_shape_invalid",
            retryable=True,
            recommended_next_action="sql_query",
            diagnostics={
                "query_language": query_language,
                "query": generated.query,
                "query_shape_issues": issues,
                "query_task_contract": contract,
                "recommended_downstream_action": recommended,
                "repair_contract": repair_contract,
            },
            validation_failure=validation_failure,
        )

    def _generation_diagnostics(
        self,
        generation: LLMQueryGenerationResult,
        *,
        previous_error: Exception | None,
    ) -> dict:
        generated = generation.generated_query
        return {
            "llm_query_generation": {
                "generation_mode": "llm",
                "query_language": generated.query_language,
                "purpose": generated.purpose,
                "expected_result_type": generated.expected_result_type,
                "selected_fields": generated.selected_fields,
                "assumptions": generated.assumptions,
                "task_coverage": generated.task_coverage,
                "query_task_contract": (
                    generated.query_task_contract.model_dump(mode="json")
                    if hasattr(generated.query_task_contract, "model_dump")
                    else generated.query_task_contract
                ),
                "query_execution": generated.query_execution.model_dump(mode="json", exclude_none=True),
                "confidence": generated.confidence,
                "repaired_from_query": generation.repaired_from_query,
                "previous_error": str(previous_error) if previous_error is not None else None,
            },
            "task_coverage": self._generation_task_coverage(generation),
        }

    def _generation_task_coverage(self, generation: LLMQueryGenerationResult) -> dict:
        generated = generation.generated_query
        coverage = generated.task_coverage if isinstance(generated.task_coverage, dict) else {}
        query_task_contract = (
            generated.query_task_contract.model_dump(mode="json")
            if hasattr(generated.query_task_contract, "model_dump")
            else generated.query_task_contract
        )
        return {
            "source": "llm_query_generation",
            "query_purpose": generated.purpose,
            "expected_result_type": generated.expected_result_type,
            "satisfied": self._string_list(coverage.get("satisfied")),
            "missing": self._coverage_missing_items(coverage),
            "next_action_hint": self._optional_string(coverage.get("next_action_hint")),
            "confidence": generated.confidence,
            "query_task_contract": query_task_contract if isinstance(query_task_contract, dict) else None,
        }

    def _coverage_missing_items(self, coverage: dict | None) -> list[str]:
        if not isinstance(coverage, dict):
            return []
        missing = self._string_list(coverage.get("missing"))
        if missing:
            return missing
        return self._string_list(coverage.get("missing_or_uncertain"))

    def _string_list(self, value) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    def _optional_string(self, value) -> str | None:
        if value in (None, ""):
            return None
        text = str(value).strip()
        return text or None

    def _infer_query_language(self, config: dict) -> str:
        db_type = str(config.get("type") or config.get("db_type") or "")
        return query_language_for_database_type(db_type)

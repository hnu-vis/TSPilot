"""Unified database query tool."""
from __future__ import annotations

import re
import time

from pydantic import BaseModel, Field, field_validator, model_validator

from app.settings import Settings
from core.database import DatabaseFactory, execute_query, normalize_query_result
from core.database.connector import DatabaseSchema
from core.database.contracts import QueryRequestContext, RenderedQuery
from core.database.dialects import dialect_for_database, query_language_for_database_type
from core.database.llm_query import LLMGeneratedQuery, LLMQueryGenerationResult, LLMQueryGenerator
from core.database.query_flow import DefaultIntentInterpreter, DefaultQueryValidator
from core.database.query_plan import QueryFilter
from core.database.repair import classify_query_error, repair_read_only_query
from core.database.schema import schema_preview
from core.database.schema_linking import SchemaLinkingPipeline
from schemas.state import RequestStateModel
from schemas.database_context import DatabaseContext
from schemas.data_fact import DataFactRequest
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
        if not (self.query and self.query.strip()) and not (self.message and self.message.strip()):
            raise ValueError("sql_query requires either message for automatic planning or query for explicit read-only execution.")
        return self


class _ExplicitQueryExecutor(BaseTool):
    """Run a safe model-authored read-only query."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._schema_linking_pipeline = SchemaLinkingPipeline()

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
        repair = repair_read_only_query(
            query=validated_input.query,
            query_language=validated_input.query_language,
        )
        query = repair.query
        self._validate_read_only(query, validated_input.query_language)
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
                result, executed_query, repair_diagnostics = await self._execute_with_repair(
                    connector=connector,
                    query=query,
                    query_language=validated_input.query_language,
                    timeout=int(validated_input.constraints.get("timeout", config.get("query_timeout", 60))),
                    initial_repair_reason=repair.reason,
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
            query=executed_query,
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
                "repair": repair_diagnostics,
            },
            **(extra_diagnostics or {}),
        }
        evidence.diagnostics["task_coverage"] = self._task_coverage_diagnostics(
            validated_input=validated_input,
            evidence=evidence.model_dump(mode="json"),
            query=executed_query,
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
        rendered = RenderedQuery(
            query_text=query,
            query_language=validated_input.query_language or self._infer_query_language(config),
        )
        contract_filters = self._contract_required_filters(validated_input.constraints)
        if contract_filters:
            missing = self._missing_rendered_required_filters(rendered, contract_filters)
            if missing:
                self._raise_missing_required_filters(
                    missing,
                    schema_linking_contract=validated_input.constraints.get("_schema_linking_contract"),
                    query=query,
                    query_language=rendered.query_language,
                )
            return

        message = (request_state.message if request_state is not None else None) or validated_input.purpose or ""
        if not message.strip():
            return
        try:
            schema = await connector.get_schema()
        except Exception:
            return
        context = QueryRequestContext(
            database_id=validated_input.database_context.database_id,
            database_type=str(config.get("type", validated_input.database_context.database_type)),
            message=message,
            constraints=validated_input.constraints,
            intent_profile=request_state.intent_profile if request_state is not None else {},
        )
        intent = DefaultIntentInterpreter().interpret(context=context)
        linking_result = self._schema_linking_pipeline.ground(
            context=context,
            schema=schema,
            intent=intent,
        )
        if not linking_result.required_filters:
            return
        validation = DefaultQueryValidator(dialect_for_database(context.database_type)).validate(
            context=context,
            plan=linking_result.plan,
            rendered_query=rendered,
        )
        missing = [issue for issue in validation.issues if issue.code == "required_filter_missing"]
        if missing:
            self._raise_missing_required_filters(
                [
                    QueryFilter(source=None, column=self._filter_column_from_message(issue.message), value=self._filter_value_from_message(issue.message))
                    for issue in missing
                ],
                schema_linking_contract=linking_result.diagnostics(),
                details="; ".join(issue.message for issue in missing),
                query=query,
                query_language=rendered.query_language,
            )

    def _contract_required_filters(self, constraints: dict) -> list[QueryFilter]:
        contract = constraints.get("_schema_linking_contract") if isinstance(constraints, dict) else None
        if not isinstance(contract, dict):
            return []
        raw_filters = contract.get("required_filters")
        if not isinstance(raw_filters, list):
            return []
        filters: list[QueryFilter] = []
        for item in raw_filters:
            if not isinstance(item, dict):
                continue
            column = str(item.get("column") or "").strip()
            if not column:
                continue
            filters.append(
                QueryFilter(
                    source=item.get("source"),
                    column=column,
                    operator=str(item.get("operator") or "="),
                    value=item.get("value"),
                )
            )
        return filters

    def _missing_rendered_required_filters(self, rendered: RenderedQuery, required_filters: list[QueryFilter]) -> list[QueryFilter]:
        dialect = dialect_for_database(rendered.query_language)
        return [
            item for item in required_filters
            if not dialect.has_rendered_filter(rendered, item)
        ]

    def _raise_missing_required_filters(
        self,
        missing: list[QueryFilter],
        *,
        schema_linking_contract: dict | None,
        details: str | None = None,
        query: str | None = None,
        query_language: str | None = None,
    ) -> None:
        rendered_details = details or "; ".join(
            f"Rendered query is missing the required filter {item.column}={item.value!r}."
            for item in missing
        )
        missing_filters = [
            {
                "source": item.source,
                "column": item.column,
                "operator": item.operator,
                "value": item.value,
            }
            for item in missing
        ]
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

    def _filter_column_from_message(self, message: str) -> str:
        match = re.search(r"filter\s+([^=\s]+)=", message)
        return match.group(1) if match else ""

    def _filter_value_from_message(self, message: str):
        match = re.search(r"=([^.;]+)", message)
        if not match:
            return None
        return match.group(1).strip().strip("'\"")

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
        from tools.query_database import QueryDatabaseTool

        self._planned_query_tool = QueryDatabaseTool(settings)
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
        config_path, config = await self._planned_query_tool._load_database_config(
            validated_input.database_context.database_id
        )
        schema, preview = await self._load_schema_and_preview(validated_input, config)
        request_state = kwargs.get("request_state")
        grounding_message = self._grounding_message(
            validated_input,
            request_state if isinstance(request_state, RequestStateModel) else None,
        )
        rule_started = time.perf_counter()
        linking_diagnostics = self._schema_linking_for_generation(
            validated_input,
            config,
            schema,
            message=grounding_message,
        )
        rule_duration_ms = int((time.perf_counter() - rule_started) * 1000)
        llm_linking_started = time.perf_counter()
        schema_linking_contract = None
        try:
            schema_linking_contract = await self._llm_query_generator.generate_schema_linking(
                database_id=validated_input.database_context.database_id,
                database_type=str(config.get("type", validated_input.database_context.database_type)),
                message=grounding_message,
                schema_preview=preview,
                rule_diagnostics=linking_diagnostics,
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
                    "rule_schema_linking": linking_diagnostics,
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
                        "schema_linking_rule_duration_ms": rule_duration_ms,
                        "schema_linking_llm_duration_ms": llm_linking_duration_ms,
                        "query_generation_duration_ms": int((time.perf_counter() - generation_started) * 1000),
                    },
                },
            ) from exc
        generation_duration_ms = int((time.perf_counter() - generation_started) * 1000)
        runtime_diagnostics = {
            "schema_linking_rule_duration_ms": rule_duration_ms,
            "schema_linking_llm_duration_ms": llm_linking_duration_ms,
            "query_generation_duration_ms": generation_duration_ms,
            "sql_query_planning_duration_ms": int((time.perf_counter() - total_started) * 1000),
        }
        try:
            return await self._execute_generated_query(
                validated_input,
                config,
                generation,
                schema_linking_diagnostics=schema_linking_contract,
                extra_runtime_diagnostics=runtime_diagnostics,
                **kwargs,
            )
        except StructuredToolError as exc:
            if exc.error_type != "query_shape_invalid":
                raise
            repair_constraints = dict(validated_input.constraints or {})
            if self._shape_issue_requires_raw_series(exc):
                repair_constraints.setdefault("evidence_shape", "raw_timeseries")
            else:
                repair_constraints.setdefault("repair_strategy", "schema_grounded_query")
            repair_started = time.perf_counter()
            repair_generation = await self._llm_query_generator.generate(
                database_id=validated_input.database_context.database_id,
                database_type=str(config.get("type", validated_input.database_context.database_type)),
                message=grounding_message,
                schema_preview=preview,
                time_range=validated_input.time_range,
                constraints=repair_constraints,
                history=validated_input.history,
                fact_requests=[item.model_dump(mode="json", exclude_none=True) for item in validated_input.fact_requests],
                previous_query=generation.generated_query.query,
                error=exc.to_observation_payload(),
                request_state=request_state,
            )
            return await self._execute_generated_query(
                validated_input.model_copy(update={"constraints": repair_constraints}),
                config,
                repair_generation,
                previous_error=exc,
                schema_linking_diagnostics=schema_linking_contract,
                extra_runtime_diagnostics={
                    **runtime_diagnostics,
                    "shape_repair_generation_duration_ms": int((time.perf_counter() - repair_started) * 1000),
                },
                **kwargs,
            )

    def _shape_issue_requires_raw_series(self, exc: StructuredToolError) -> bool:
        diagnostics = exc.diagnostics if isinstance(exc.diagnostics, dict) else {}
        issues = diagnostics.get("query_shape_issues") if isinstance(diagnostics.get("query_shape_issues"), list) else []
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            recommended = str(issue.get("recommended_shape") or "").strip().lower()
            if recommended == "schema_grounded_flux_aggregate":
                return False
            if recommended in {"raw_series", "raw_series_or_simple_aggregate_table"}:
                return True
        return True

    async def _load_schema_and_preview(self, validated_input: SqlQueryInput, config: dict) -> tuple[DatabaseSchema | None, dict]:
        reference_dataset = config.get("reference_dataset")
        if isinstance(reference_dataset, dict):
            return None, self._reference_dataset_schema_preview(config)
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

    def _schema_linking_for_generation(
        self,
        validated_input: SqlQueryInput,
        config: dict,
        schema: DatabaseSchema | None,
        *,
        message: str | None = None,
    ) -> dict | None:
        if schema is None:
            return None
        message = message if message is not None else validated_input.message or ""
        if not message.strip():
            return None
        context = QueryRequestContext(
            database_id=validated_input.database_context.database_id,
            database_type=str(config.get("type", validated_input.database_context.database_type)),
            message=message,
            time_range=validated_input.time_range,
            constraints=validated_input.constraints,
            history=validated_input.history,
            intent_profile=validated_input.intent_profile,
        )
        intent = DefaultIntentInterpreter().interpret(context=context)
        return SchemaLinkingPipeline().ground(
            context=context,
            schema=schema,
            intent=intent,
        ).diagnostics()

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

    def _reference_dataset_schema_preview(self, config: dict) -> dict:
        reference_dataset = config.get("reference_dataset") if isinstance(config.get("reference_dataset"), dict) else {}
        table_name = (
            reference_dataset.get("measurement")
            or reference_dataset.get("metric_name")
            or reference_dataset.get("table")
            or reference_dataset.get("series_name")
            or config.get("database")
            or config.get("name")
        )
        fields = list(reference_dataset.get("field_columns") or [])
        value_column = reference_dataset.get("value_column")
        if value_column and value_column not in fields:
            fields.append(value_column)
        time_column = reference_dataset.get("timestamp_column")
        return {
            "source": "reference_dataset_config",
            "database_type": config.get("type"),
            "query_language": self._infer_query_language(config),
            "tables_or_measurements": [
                {
                    "name": table_name,
                    "time_column": time_column,
                    "field_columns": fields[:100],
                    "time_range": reference_dataset.get("time_range"),
                    "sample_rows": reference_dataset.get("sample_rows", [])[:3]
                    if isinstance(reference_dataset.get("sample_rows"), list)
                    else [],
                }
            ],
        }

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
        if isinstance(config.get("reference_dataset"), dict):
            return await self._execute_reference_dataset_query(validated_input, config, generated, generation, previous_error, **kwargs)
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
                    },
                    fact_requests=validated_input.fact_requests,
                ),
                mode="llm",
                extra_metadata={
                    "generation_mode": "llm",
                    "expected_result_type": generated.expected_result_type,
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
        if issues:
            repair = dialect.repair_query(
                query=generated.query,
                query_language=query_language,
                error={"query_shape_issues": issues},
            )
            if repair.changed:
                generated.query = repair.query
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

    async def _execute_reference_dataset_query(
        self,
        validated_input: SqlQueryInput,
        config: dict,
        generated: LLMGeneratedQuery,
        generation: LLMQueryGenerationResult,
        previous_error: Exception | None,
        **kwargs,
    ) -> dict:
        from tools.query_database import QueryDatabaseInput

        self._explicit_query_executor._validate_read_only(
            generated.query,
            generated.query_language or self._infer_query_language(config),
        )
        constraints = {
            **validated_input.constraints,
            "selected_fields": generated.selected_fields,
            "expected_result_type": generated.expected_result_type or "timeseries",
            "llm_generated_query": generated.query,
            "llm_query_language": generated.query_language or self._infer_query_language(config),
        }
        evidence = await self._planned_query_tool.execute(
            QueryDatabaseInput(
                message=validated_input.message or "",
                database_context=validated_input.database_context,
                time_range=validated_input.time_range,
                constraints=constraints,
                intent_profile=validated_input.intent_profile,
                selected_database=validated_input.selected_database,
                selected_database_type=validated_input.selected_database_type,
                history=validated_input.history,
            ),
            **kwargs,
        )
        evidence["query"] = generated.query
        evidence["query_language"] = generated.query_language or self._infer_query_language(config)
        evidence["metadata"] = {
            **evidence.get("metadata", {}),
            "sql_query_mode": "llm",
            "generation_mode": "llm",
            "purpose": generated.purpose,
            "expected_result_type": generated.expected_result_type,
        }
        evidence["diagnostics"] = {
            **evidence.get("diagnostics", {}),
            **self._generation_diagnostics(
                generation,
                previous_error=previous_error,
                reference_dataset=True,
            ),
        }
        return evidence

    def _generation_diagnostics(
        self,
        generation: LLMQueryGenerationResult,
        *,
        previous_error: Exception | None,
        reference_dataset: bool = False,
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
                "confidence": generated.confidence,
                "repaired_from_query": generation.repaired_from_query,
                "previous_error": str(previous_error) if previous_error is not None else None,
                "reference_dataset_execution": reference_dataset,
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

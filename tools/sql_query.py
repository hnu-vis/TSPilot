"""Unified database query tool."""
from __future__ import annotations

import re

from pydantic import BaseModel, Field, model_validator

from app.settings import Settings
from core.database import DatabaseFactory, execute_query, normalize_query_result
from core.database.connector import DatabaseSchema
from core.database.contracts import QueryRequestContext, RenderedQuery
from core.database.llm_query import LLMGeneratedQuery, LLMQueryGenerationResult, LLMQueryGenerator
from core.database.query_flow import DefaultIntentInterpreter, DefaultQueryValidator
from core.database.repair import classify_query_error, repair_read_only_query
from core.database.schema import schema_preview
from core.database.schema_linking import SchemaLinkingPipeline
from schemas.state import RequestStateModel
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

    async def _validate_required_filters(
        self,
        *,
        connector,
        config: dict,
        validated_input: _ExplicitQueryInput,
        query: str,
        request_state: RequestStateModel | None,
    ) -> None:
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
        rendered = RenderedQuery(
            query_text=query,
            query_language=validated_input.query_language or self._infer_query_language(config),
        )
        validation = DefaultQueryValidator().validate(
            context=context,
            plan=linking_result.plan,
            rendered_query=rendered,
        )
        missing = [issue for issue in validation.issues if issue.code == "required_filter_missing"]
        if missing:
            details = "; ".join(issue.message for issue in missing)
            raise ValueError(
                "Explicit query is missing filters required by the user request. "
                f"{details} Preserve those filters or use sql_query automatic planning."
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
        value_aliases = {"value", "_value", "metric_value"}
        if actual & value_aliases:
            return []
        missing_fields = sorted(
            field
            for field in expected
            if not self._field_present_in_columns(field, actual)
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
        normalized_query = str(query or "").lower()
        if str(query_language or "").lower() == "flux":
            has_raw_limit = "limit(" in normalized_query and "aggregatewindow" not in normalized_query
        else:
            has_raw_limit = bool(re.search(r"\blimit\s+\d+\b", normalized_query, flags=re.IGNORECASE))
        if not has_raw_limit:
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

    def _field_present_in_columns(self, field: str, columns: set[str]) -> bool:
        time_aliases = {"time", "_time", "timestamp", "datetime", "date"}
        if field in time_aliases and columns & time_aliases:
            return True
        if field in columns:
            return True
        normalized_field = field.replace("-", "_").strip("_")
        for column in columns:
            normalized_column = column.replace("-", "_").strip("_")
            tokens = [token for token in normalized_column.split("_") if token]
            if normalized_field in tokens:
                return True
            if normalized_column.endswith(f"_{normalized_field}"):
                return True
        return False

    def _projected_columns_from_query(self, *, query: str | None, query_language: str | None) -> set[str]:
        if not query or str(query_language or "").lower() != "flux":
            return set()
        projected: set[str] = set()
        for match in re.finditer(r"keep\s*\(\s*columns\s*:\s*\[([^\]]*)\]", query, flags=re.IGNORECASE | re.DOTALL):
            for item in re.findall(r'"([^"]+)"|\'([^\']+)\'', match.group(1)):
                column = (item[0] or item[1]).strip().lower()
                if column:
                    projected.add(column)
        return self._apply_flux_renames(projected, query)

    def _apply_flux_renames(self, projected: set[str], query: str) -> set[str]:
        if not projected:
            return projected
        aliases: dict[str, str] = {}
        for match in re.finditer(r"rename\s*\(\s*columns\s*:\s*\{([^}]*)\}", query, flags=re.IGNORECASE | re.DOTALL):
            for quoted_source, quoted_target, bare_source, bare_target in re.findall(
                r'"([^"]+)"\s*:\s*"([^"]+)"|([A-Za-z_][\w]*)\s*:\s*"([^"]+)"',
                match.group(1),
            ):
                source_name = (quoted_source or bare_source).strip().lower()
                target_name = (quoted_target or bare_target).strip().lower()
                if source_name and target_name:
                    aliases[source_name] = target_name
        if not aliases:
            return projected
        return {aliases.get(column, column) for column in projected}

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
        if validated_input.query and validated_input.query.strip():
            return await self._explicit_query_executor.execute_query_input(
                _ExplicitQueryInput(
                    database_context=validated_input.database_context,
                    query=validated_input.query,
                    query_language=validated_input.query_language,
                    purpose=validated_input.purpose,
                    constraints=validated_input.constraints,
                ),
                mode="explicit",
                **kwargs,
            )

        return await self._execute_llm_planned_query(validated_input, **kwargs)

    async def _execute_llm_planned_query(self, validated_input: SqlQueryInput, **kwargs) -> dict:
        if self._llm_query_generator is None:
            raise RuntimeError("sql_query automatic mode requires an LLM query generator.")

        config_path, config = await self._planned_query_tool._load_database_config(
            validated_input.database_context.database_id
        )
        schema, preview = await self._load_schema_and_preview(validated_input, config)
        linking_diagnostics = self._schema_linking_for_generation(validated_input, config, schema)
        if linking_diagnostics:
            preview = {**preview, "schema_linking": linking_diagnostics}
        generation = await self._llm_query_generator.generate(
            database_id=validated_input.database_context.database_id,
            database_type=str(config.get("type", validated_input.database_context.database_type)),
            message=validated_input.message or "",
            schema_preview=preview,
            time_range=validated_input.time_range,
            constraints=validated_input.constraints,
            history=validated_input.history,
            request_state=kwargs.get("request_state"),
        )
        try:
            return await self._execute_generated_query(
                validated_input,
                config,
                generation,
                schema_linking_diagnostics=linking_diagnostics,
                **kwargs,
            )
        except ValueError:
            raise
        except Exception as exc:
            repair_generation = await self._llm_query_generator.generate(
                database_id=validated_input.database_context.database_id,
                database_type=str(config.get("type", validated_input.database_context.database_type)),
                message=validated_input.message or "",
                schema_preview=preview,
                time_range=validated_input.time_range,
                constraints=validated_input.constraints,
                history=validated_input.history,
                previous_query=generation.generated_query.query,
                error=exc,
                request_state=kwargs.get("request_state"),
            )
            return await self._execute_generated_query(
                validated_input,
                config,
                repair_generation,
                previous_error=exc,
                schema_linking_diagnostics=linking_diagnostics,
                **kwargs,
            )

    async def _load_schema_and_preview(self, validated_input: SqlQueryInput, config: dict) -> tuple[DatabaseSchema | None, dict]:
        reference_dataset = config.get("reference_dataset")
        if isinstance(reference_dataset, dict):
            return None, self._reference_dataset_schema_preview(config)
        try:
            connector = await DatabaseFactory.create_connector(**config)
            async with connector:
                schema = await connector.get_schema()
            return schema, schema_preview(schema)
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
    ) -> dict | None:
        if schema is None:
            return None
        message = validated_input.message or ""
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
        **kwargs,
    ) -> dict:
        generated = generation.generated_query
        if isinstance(config.get("reference_dataset"), dict):
            return await self._execute_reference_dataset_query(validated_input, config, generated, generation, previous_error, **kwargs)
        return await self._explicit_query_executor.execute_query_input(
            _ExplicitQueryInput(
                database_context=validated_input.database_context,
                query=generated.query,
                query_language=generated.query_language or self._infer_query_language(config),
                purpose=generated.purpose,
                constraints=validated_input.constraints,
            ),
            mode="llm",
            extra_metadata={
                "generation_mode": "llm",
                "expected_result_type": generated.expected_result_type,
            },
            extra_diagnostics={
                **self._generation_diagnostics(generation, previous_error=previous_error),
                **({"schema_linking_generation": schema_linking_diagnostics} if schema_linking_diagnostics else {}),
            },
            **kwargs,
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
        return {
            "source": "llm_query_generation",
            "query_purpose": generated.purpose,
            "expected_result_type": generated.expected_result_type,
            "satisfied": self._string_list(coverage.get("satisfied")),
            "missing": self._coverage_missing_items(coverage),
            "next_action_hint": self._optional_string(coverage.get("next_action_hint")),
            "confidence": generated.confidence,
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
        if db_type == "influxdb":
            return "flux"
        if db_type == "prometheus":
            return "promql"
        return "sql"

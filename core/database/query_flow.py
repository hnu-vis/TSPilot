"""Backend-agnostic query planning and rendering flow."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from core.time_range import normalize_time_value, parse_time_to_utc
from schemas.database import DatabaseEvidence

from .connector import DatabaseSchema, QueryResult
from .contracts import (
    DialectRenderer,
    EvidenceNormalizer,
    FieldMapper,
    FieldMappingCandidate,
    IntentInterpreter,
    LogicalQueryPlanner,
    QueryExecutionTrace,
    QueryIntent,
    QueryRepairDecision,
    QueryRepairPolicy,
    QueryArtifactRef,
    QueryRequestContext,
    QueryResultSnapshotStore,
    QueryValidationReport,
    QueryValidator,
    RenderedQuery,
    SchemaCatalog,
    ValidationIssue,
    ValueDomainProbe,
)
from .engine import infer_evidence_family, normalize_query_result
from .query_compiler import QueryCompiler
from .query_plan import DatabaseQueryPlan, TimeRangePlan
from .schema_linking import SchemaLinkingPipeline


class ConnectorSchemaCatalog(SchemaCatalog):
    """Load schema through a live connector."""

    def __init__(self, connector):
        self._connector = connector

    async def load_schema(self, *, context: QueryRequestContext) -> DatabaseSchema:
        return await self._connector.get_schema()


class DefaultIntentInterpreter(IntentInterpreter):
    """Deterministically map the request to evidence and query shape."""

    _AGGREGATION_KEYWORDS = {
        "avg": "avg",
        "average": "avg",
        "mean": "avg",
        "平均": "avg",
        "均值": "avg",
        "max": "max",
        "最大": "max",
        "min": "min",
        "最小": "min",
        "sum": "sum",
        "总和": "sum",
        "count": "count",
        "计数": "count",
        "数量": "count",
        "总数": "count",
        "总条数": "count",
        "多少条": "count",
        "几条": "count",
    }

    _FACT_KEYWORDS = {
        "seasonality": ("周期", "周期性", "每天", "每周", "seasonality", "daily", "weekly"),
        "trend": ("趋势", "走势", "trend"),
        "outlier": ("异常", "离群", "anomaly", "outlier"),
        "forecast": ("预测", "forecast"),
        "difference": ("变化", "增减", "difference", "change"),
        "extreme": ("最高", "最低", "最大", "最小", "peak", "trough", "extreme", "extrema", "maximum", "minimum"),
        "aggregation": ("平均", "均值", "总和", "最大", "最小", "count", "sum", "avg", "mean"),
    }

    _RAW_SERIES_FACTS = {"seasonality", "trend", "outlier", "forecast"}
    _NEGATED_AGGREGATION_PATTERNS = (
        "不要使用 max",
        "不要用 max",
        "avoid max",
        "avoid aggregation",
        "avoid aggregations",
        "avoid aggregate",
        "not use max",
        "without max",
    )
    def interpret(self, *, context: QueryRequestContext) -> QueryIntent:
        intent_profile = context.intent_profile or {}
        profile_fact_types = [
            str(item)
            for item in intent_profile.get("requested_capabilities", [])
            if item
        ]
        normalized = context.message.lower()
        fact_families = profile_fact_types or self._detect_fact_families(normalized)
        evidence_family = infer_evidence_family(context.message)
        aggregation = self._detect_aggregation(normalized, context.constraints, fact_families)
        if not aggregation:
            aggregation = self._aggregation_from_profile(intent_profile, fact_families)
        if intent_profile.get("analysis_kind") == "statistical_summary" or aggregation:
            evidence_family = "statistics"
        query_shape = "raw_timeseries"
        requires_raw_points = evidence_family == "timeseries"
        requires_full_fidelity = any(fact in self._RAW_SERIES_FACTS for fact in fact_families)
        if requires_full_fidelity and not aggregation:
            evidence_family = "timeseries"
        if evidence_family == "statistics":
            query_shape = "scalar_aggregate"
            requires_raw_points = False
        elif evidence_family == "table":
            query_shape = "row_detail"
            requires_raw_points = False
        elif evidence_family in {"schema", "metric_list"}:
            query_shape = evidence_family
            requires_raw_points = False
        notes = []
        if aggregation and query_shape == "raw_timeseries":
            notes.append("Aggregation keywords detected but raw time-series evidence is preferred for analysis fidelity.")
        return QueryIntent(
            requested_fact_families=fact_families,
            evidence_family=evidence_family,
            query_shape=query_shape,
            notes=notes,
            requires_raw_points=requires_raw_points,
            requires_full_fidelity=requires_full_fidelity,
            filters={"aggregation": aggregation} if aggregation else {},
        )

    def _aggregation_from_profile(
        self,
        intent_profile: dict[str, Any],
        fact_families: list[str],
    ) -> str | None:
        metrics = [str(item).lower() for item in intent_profile.get("requested_metrics", []) if item]
        if "max" in metrics or "maximum" in metrics or "max_or_min" in metrics:
            return "max"
        if "min" in metrics or "minimum" in metrics:
            return "min"
        if "average" in metrics or "avg" in metrics or "mean" in metrics or "aggregate" in metrics:
            return "avg"
        if "extreme" in fact_families:
            return "max"
        if "aggregation" in fact_families:
            return "avg"
        return None

    def _detect_aggregation(
        self,
        normalized: str,
        constraints: dict[str, Any],
        fact_families: list[str],
    ) -> str | None:
        if any(fact in self._RAW_SERIES_FACTS for fact in fact_families):
            return None
        if any(pattern in normalized for pattern in self._NEGATED_AGGREGATION_PATTERNS):
            return None
        if any(token in normalized for token in ("max_points", "avoid_aggregations", "avoid_aggregates")):
            return None
        avoid_aggregations = constraints.get("avoid_aggregations") or constraints.get("avoid_aggregates") or []
        if avoid_aggregations:
            return None
        for keyword, aggregation in self._AGGREGATION_KEYWORDS.items():
            if keyword in normalized:
                return aggregation
        return None

    def _detect_fact_families(self, normalized: str) -> list[str]:
        families: list[str] = []
        for fact_family, keywords in self._FACT_KEYWORDS.items():
            if any(keyword in normalized for keyword in keywords):
                families.append(fact_family)
        return families


class DefaultFieldMapper(FieldMapper):
    """Ground source and field candidates from schema metadata."""

    def __init__(self):
        self._pipeline = SchemaLinkingPipeline()

    def map_fields(
        self,
        *,
        context: QueryRequestContext,
        schema: DatabaseSchema,
        intent: QueryIntent,
    ) -> list[FieldMappingCandidate]:
        return self._pipeline.map_fields(context=context, schema=schema, intent=intent)


class DefaultLogicalQueryPlanner(LogicalQueryPlanner):
    """Build a conservative logical plan from intent and grounded fields."""

    def __init__(self):
        self._pipeline = SchemaLinkingPipeline()

    def build_plan(
        self,
        *,
        context: QueryRequestContext,
        schema: DatabaseSchema,
        intent: QueryIntent,
        field_mappings: list[FieldMappingCandidate],
    ) -> DatabaseQueryPlan:
        return self._pipeline.build_plan(
            context=context,
            schema=schema,
            intent=intent,
            field_mappings=field_mappings,
        )


class CompositeDialectRenderer(DialectRenderer):
    """Render logical plans into SQL, Flux, or PromQL."""

    _SQL_DIALECTS = {"timescaledb", "postgresql", "questdb", "clickhouse", "sql"}

    def __init__(self, config: dict[str, Any]):
        self._config = config
        self._compiler = QueryCompiler()

    def render(
        self,
        *,
        context: QueryRequestContext,
        plan: DatabaseQueryPlan,
    ) -> RenderedQuery:
        db_type = context.database_type.lower()
        if db_type in self._SQL_DIALECTS:
            compiled = self._compiler.compile(plan, db_type=db_type, dialect=db_type)
            return RenderedQuery(
                query_text=compiled.query,
                query_language=compiled.language,
                warnings=list(compiled.warnings),
            )
        if db_type == "influxdb":
            return self._render_flux(plan, context)
        if db_type == "prometheus":
            return self._render_promql(plan, context)
        compiled = self._compiler.compile(plan, db_type=db_type, dialect=db_type)
        return RenderedQuery(
            query_text=compiled.query,
            query_language=compiled.language,
            warnings=list(compiled.warnings),
        )

    def _render_flux(self, plan: DatabaseQueryPlan, context: QueryRequestContext) -> RenderedQuery:
        bucket = str(self._config.get("bucket") or self._config.get("database") or "")
        source = plan.sources[0] if plan.sources else None
        measurement = source.name if source else ""
        field_projection = next((item for item in plan.projections if item.alias == "value"), None)
        field_name = field_projection.column if field_projection else (source.value_columns[0] if source and source.value_columns else "_value")
        range_clause = self._flux_range(plan.time_range)
        lines = [f'from(bucket: "{bucket}")', f"  |> {range_clause}"]
        if measurement:
            lines.append(f'  |> filter(fn: (r) => r._measurement == "{measurement}")')
        if field_name:
            lines.append(f'  |> filter(fn: (r) => r._field == "{field_name}")')
        for item in plan.filters:
            if item.column in {"_measurement", "_field", "_time", "time", "timestamp"}:
                continue
            lines.append(
                f'  |> filter(fn: (r) => r.{item.column} {self._flux_operator(item.operator)} "{self._escape_flux_string(item.value)}")'
            )
        aggregation = next((item.aggregation for item in plan.projections if item.aggregation), None)
        if aggregation:
            flux_fn = {"avg": "mean", "mean": "mean", "max": "max", "min": "min", "sum": "sum", "count": "count"}.get(aggregation, aggregation)
            lines.append("  |> group()")
            lines.append(f"  |> {flux_fn}()")
        return RenderedQuery(query_text="\n".join(lines), query_language="flux")

    def _render_promql(self, plan: DatabaseQueryPlan, context: QueryRequestContext) -> RenderedQuery:
        source = plan.sources[0] if plan.sources else None
        metric_name = source.name if source else ""
        label_filters = [
            item for item in plan.filters
            if item.column not in {"timestamp", "time", "_time"}
        ]
        matcher_text = ""
        if label_filters:
            matcher_text = "{" + ",".join(f'{item.column}="{item.value}"' for item in label_filters) + "}"
        query_text = f"{metric_name}{matcher_text}".strip()
        return RenderedQuery(query_text=query_text, query_language="promql")

    def _flux_range(self, time_range: TimeRangePlan) -> str:
        if time_range.start and time_range.end:
            return f'range(start: {self._flux_time(time_range.start)}, stop: {self._flux_time(time_range.end)})'
        if time_range.start:
            return f'range(start: {self._flux_time(time_range.start)})'
        if time_range.lookback:
            return f"range(start: -{time_range.lookback})"
        default_range = self._default_flux_time_range()
        if default_range.get("start") and default_range.get("end"):
            return f'range(start: {self._flux_time(default_range["start"])}, stop: {self._flux_time(default_range["end"])})'
        if default_range.get("start"):
            return f'range(start: {self._flux_time(default_range["start"])})'
        return "range(start: 1970-01-01T00:00:00Z)"

    def _default_flux_time_range(self) -> dict[str, Any]:
        configured = self._config.get("default_query_time_range") or self._config.get("default_time_range")
        if isinstance(configured, dict):
            return normalize_time_range(configured) or {}
        reference_dataset = self._config.get("reference_dataset")
        if isinstance(reference_dataset, dict):
            configured = reference_dataset.get("time_range")
            if isinstance(configured, dict):
                return normalize_time_range(configured) or {}
        return {"start": "1970-01-01T00:00:00Z"}

    def _flux_time(self, value: str) -> str:
        return normalize_time_value(value)

    def _flux_operator(self, operator: str) -> str:
        return "==" if operator == "=" else operator

    def _escape_flux_string(self, value: Any) -> str:
        return str(value).replace("\\", "\\\\").replace('"', '\\"')


class DefaultQueryValidator(QueryValidator):
    """Validate that rendering preserved key plan semantics."""

    def validate(
        self,
        *,
        context: QueryRequestContext,
        plan: DatabaseQueryPlan,
        rendered_query: RenderedQuery,
        result: QueryResult | None = None,
    ) -> QueryValidationReport:
        issues: list[ValidationIssue] = []
        text = rendered_query.query_text.lower()
        if context.time_range and rendered_query.query_language in {"sql", "timescaledb", "postgresql", "questdb", "clickhouse", "flux"}:
            start = str(context.time_range.get("start") or "").lower()
            end = str(context.time_range.get("end") or "").lower()
            if start and start not in text:
                issues.append(ValidationIssue(code="time_range_start_missing", message="Rendered query is missing the absolute start time."))
            if end and rendered_query.query_language == "flux" and end not in text:
                issues.append(ValidationIssue(code="time_range_end_missing", message="Rendered query is missing the absolute end time."))
        if plan.output_shape == "long_series" and any(token in text for token in ["mean()", "avg(", "sum(", "count(", "max(", "min("]):
            if not any(item.aggregation for item in plan.projections):
                issues.append(ValidationIssue(code="unexpected_aggregation", message="Raw time-series plan was rendered with an aggregate function."))
        if result is not None and result.row_count == 0:
            issues.append(ValidationIssue(code="empty_result", message="Query executed successfully but returned no rows.", severity="warning"))
        for item in plan.filters:
            if item.column in {"_time", "time", "timestamp"}:
                continue
            if not self._has_rendered_filter(rendered_query, item):
                issues.append(
                    ValidationIssue(
                        code="required_filter_missing",
                        message=f"Rendered query is missing the required filter {item.column}={item.value!r}.",
                    )
                )
        safe_to_repair = any(
            issue.code in {"time_range_start_missing", "time_range_end_missing", "unexpected_aggregation", "required_filter_missing"}
            for issue in issues
        )
        return QueryValidationReport(valid=not any(issue.severity == "error" for issue in issues), issues=issues, safe_to_repair=safe_to_repair)

    def _has_rendered_filter(self, rendered_query: RenderedQuery, item: QueryFilter) -> bool:
        text = rendered_query.query_text.lower()
        column = str(item.column).lower()
        value = str(item.value).lower()
        if rendered_query.query_language == "flux":
            return f"r.{column}" in text and f'"{value}"' in text
        if rendered_query.query_language == "promql":
            return column in text and f'"{value}"' in text
        return column in text and value in text


class DefaultQueryRepairPolicy(QueryRepairPolicy):
    """Only repair missing time ranges or accidental aggregation."""

    def __init__(self, renderer: CompositeDialectRenderer):
        self._renderer = renderer

    def decide(
        self,
        *,
        context: QueryRequestContext,
        plan: DatabaseQueryPlan,
        rendered_query: RenderedQuery,
        validation: QueryValidationReport,
    ) -> QueryRepairDecision:
        if not validation.safe_to_repair:
            return QueryRepairDecision()
        if any(issue.code == "unexpected_aggregation" for issue in validation.issues):
            repaired = self._renderer.render(context=context, plan=plan)
            return QueryRepairDecision(should_retry=True, reason="Re-rendered the raw time-series plan without aggregation.", replacement_query=repaired)
        if any(issue.code in {"time_range_start_missing", "time_range_end_missing", "required_filter_missing"} for issue in validation.issues):
            repaired = self._renderer.render(context=context, plan=plan)
            return QueryRepairDecision(should_retry=True, reason="Re-rendered the query to preserve explicit constraints and filters.", replacement_query=repaired)
        return QueryRepairDecision()


class DefaultEvidenceNormalizer(EvidenceNormalizer):
    """Normalize backend results and attach query trace diagnostics."""

    def normalize(
        self,
        *,
        context: QueryRequestContext,
        rendered_query: RenderedQuery,
        result: QueryResult,
        plan: DatabaseQueryPlan,
    ) -> DatabaseEvidence:
        evidence = normalize_query_result(
            database_id=context.database_id,
            database_type=context.database_type,
            query_language=rendered_query.query_language,
            query=rendered_query.query_text,
            result=result,
        )
        return evidence


class FileQueryResultSnapshotStore(QueryResultSnapshotStore):
    """Persist full query results into a local snapshot directory."""

    def __init__(self, root_dir: str | Path):
        self._root_dir = Path(root_dir)
        self._root_dir.mkdir(parents=True, exist_ok=True)

    def store(
        self,
        *,
        context: QueryRequestContext,
        rendered_query: RenderedQuery,
        result: QueryResult,
    ) -> QueryArtifactRef:
        query_hash = hashlib.sha1(rendered_query.query_text.encode("utf-8")).hexdigest()[:12]
        artifact_id = f"qry_{context.database_id}_{query_hash}"
        payload = {
            "artifact_id": artifact_id,
            "artifact_kind": "query_result_snapshot",
            "database_id": context.database_id,
            "database_type": context.database_type,
            "query_language": rendered_query.query_language,
            "query_text": rendered_query.query_text,
            "structured_request": rendered_query.structured_request,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "time_range": context.time_range,
            "constraints": context.constraints,
            "result": {
                "columns": result.columns,
                "rows": result.rows,
                "row_count": result.row_count,
                "execution_time_ms": result.execution_time_ms,
                "truncated": result.truncated,
            },
        }
        path = self._root_dir / f"{artifact_id}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return QueryArtifactRef(
            artifact_id=artifact_id,
            artifact_kind="query_result_snapshot",
            uri=str(path),
            metadata={
                "row_count": result.row_count,
                "query_language": rendered_query.query_language,
            },
        )


class DatabaseQueryFlow:
    """End-to-end database query flow orchestration."""

    def __init__(
        self,
        *,
        connector,
        config: dict[str, Any],
        schema_catalog: SchemaCatalog | None = None,
        intent_interpreter: IntentInterpreter | None = None,
        field_mapper: FieldMapper | None = None,
        planner: LogicalQueryPlanner | None = None,
        renderer: DialectRenderer | None = None,
        validator: QueryValidator | None = None,
        repair_policy: QueryRepairPolicy | None = None,
        snapshot_store: QueryResultSnapshotStore | None = None,
        normalizer: EvidenceNormalizer | None = None,
        schema_linking_pipeline: SchemaLinkingPipeline | None = None,
    ):
        self._connector = connector
        self._config = config
        self._schema_catalog = schema_catalog or ConnectorSchemaCatalog(connector)
        self._intent_interpreter = intent_interpreter or DefaultIntentInterpreter()
        self._field_mapper = field_mapper or DefaultFieldMapper()
        self._planner = planner or DefaultLogicalQueryPlanner()
        self._schema_linking_pipeline = schema_linking_pipeline or SchemaLinkingPipeline()
        self._use_default_schema_linking = field_mapper is None and planner is None
        self._renderer = renderer or CompositeDialectRenderer(config)
        self._validator = validator or DefaultQueryValidator()
        self._repair_policy = repair_policy or DefaultQueryRepairPolicy(self._renderer)
        snapshot_dir = config.get("snapshot_dir", "cache_data/query_snapshots")
        self._snapshot_store = snapshot_store or FileQueryResultSnapshotStore(snapshot_dir)
        self._normalizer = normalizer or DefaultEvidenceNormalizer()

    async def run(self, *, context: QueryRequestContext, execute_range_query_fn=None) -> DatabaseEvidence:
        schema = await self._schema_catalog.load_schema(context=context)
        intent = self._intent_interpreter.interpret(context=context)
        if self._use_default_schema_linking:
            linking_result = self._schema_linking_pipeline.ground(context=context, schema=schema, intent=intent)
            field_mappings = linking_result.field_mappings
            plan = linking_result.plan
        else:
            field_mappings = self._field_mapper.map_fields(context=context, schema=schema, intent=intent)
            plan = self._planner.build_plan(context=context, schema=schema, intent=intent, field_mappings=field_mappings)
        schema, field_mappings, plan = await self._maybe_probe_value_domains(
            context=context,
            schema=schema,
            intent=intent,
            field_mappings=field_mappings,
            plan=plan,
        )
        rendered_query = self._renderer.render(context=context, plan=plan)
        validation = self._validator.validate(context=context, plan=plan, rendered_query=rendered_query)
        repaired = False
        if not validation.valid:
            decision = self._repair_policy.decide(
                context=context,
                plan=plan,
                rendered_query=rendered_query,
                validation=validation,
            )
            if decision.should_retry and decision.replacement_query is not None:
                rendered_query = decision.replacement_query
                repaired = True
        result = await self._execute(context=context, intent=intent, rendered_query=rendered_query, execute_range_query_fn=execute_range_query_fn)
        snapshot_ref = self._snapshot_store.store(
            context=context,
            rendered_query=rendered_query,
            result=result,
        )
        result_validation = self._validator.validate(
            context=context,
            plan=plan,
            rendered_query=rendered_query,
            result=result,
        )
        evidence = self._normalizer.normalize(
            context=context,
            rendered_query=rendered_query,
            result=result,
            plan=plan,
        )
        trace = QueryExecutionTrace(
            adapter_type=context.database_type,
            logical_plan=plan.to_dict(),
            rendered_query={
                "query_text": rendered_query.query_text,
                "query_language": rendered_query.query_language,
                "structured_request": rendered_query.structured_request,
                "warnings": rendered_query.warnings,
            },
            schema_summary={
                "table_count": len(schema.tables),
                "tables": [table.name for table in schema.tables[:8]],
            },
            field_mappings=[
                {
                    "user_term": item.user_term,
                    "source_name": item.source_name,
                    "field_name": item.field_name,
                    "role": item.role,
                    "confidence": item.confidence,
                    "evidence": item.evidence,
                }
                for item in field_mappings[:16]
            ],
            snapshot_ref=asdict(snapshot_ref),
            repaired=repaired,
            validation_issues=[asdict(issue) for issue in [*validation.issues, *result_validation.issues]],
            raw_result_summary={
                "row_count": result.row_count,
                "columns": result.columns,
                "execution_time_ms": result.execution_time_ms,
                "truncated": result.truncated,
            },
        )
        evidence.diagnostics = {
            **evidence.diagnostics,
            "query_trace": asdict(trace),
            "query_snapshot_ref": asdict(snapshot_ref),
        }
        evidence.metadata = {
            **evidence.metadata,
            "evidence_family": intent.evidence_family,
            "query_shape": intent.query_shape,
            "requested_fact_families": intent.requested_fact_families,
        }
        return evidence

    async def _maybe_probe_value_domains(
        self,
        *,
        context: QueryRequestContext,
        schema: DatabaseSchema,
        intent: QueryIntent,
        field_mappings: list[FieldMappingCandidate],
        plan: DatabaseQueryPlan,
    ) -> tuple[DatabaseSchema, list[FieldMappingCandidate], DatabaseQueryPlan]:
        if not isinstance(self._connector, ValueDomainProbe):
            return schema, field_mappings, plan
        if plan.filters or not isinstance(plan.schema_linking, dict):
            return schema, field_mappings, plan
        linked_sources = list(plan.schema_linking.get("sources") or [])
        if not linked_sources:
            return schema, field_mappings, plan
        source_payload = linked_sources[0]
        source_name = str(source_payload.get("name") or "")
        if not source_name:
            return schema, field_mappings, plan
        dimension_columns = [
            str(column)
            for column in list(source_payload.get("dimension_columns") or [])
            if column not in {"_measurement", "_field", "_time", "time", "timestamp", "result", "table"}
        ]
        if not dimension_columns:
            return schema, field_mappings, plan
        value_domains = schema.metadata.setdefault("value_domains", {})
        existing_domains = value_domains.get(source_name)
        if isinstance(existing_domains, dict) and any(existing_domains.get(column) for column in dimension_columns):
            return schema, field_mappings, plan
        probed = await self._connector.probe_value_domains(
            source_name=source_name,
            columns=dimension_columns[:4],
            limit=100,
        )
        if not probed:
            return schema, field_mappings, plan
        merged_domains = value_domains.setdefault(source_name, {})
        for column, values in probed.items():
            existing_values = merged_domains.setdefault(str(column), [])
            for value in values:
                normalized = str(value)
                if normalized not in existing_values:
                    existing_values.append(normalized)
        if self._use_default_schema_linking:
            linking_result = self._schema_linking_pipeline.ground(context=context, schema=schema, intent=intent)
            field_mappings = linking_result.field_mappings
            plan = linking_result.plan
        else:
            field_mappings = self._field_mapper.map_fields(context=context, schema=schema, intent=intent)
            plan = self._planner.build_plan(context=context, schema=schema, intent=intent, field_mappings=field_mappings)
        plan.notes.append("value_domains_probed=true")
        return schema, field_mappings, plan

    async def _execute(self, *, context: QueryRequestContext, intent: QueryIntent, rendered_query: RenderedQuery, execute_range_query_fn=None) -> QueryResult:
        if (
            context.database_type.lower() == "prometheus"
            and context.time_range
            and intent.query_shape == "raw_timeseries"
            and hasattr(self._connector, "get_range")
            and execute_range_query_fn is not None
        ):
            step = self._prometheus_step(context.time_range, context.constraints)
            return await execute_range_query_fn(
                self._connector,
                rendered_query.query_text,
                start=self._parse_time(context.time_range["start"]),
                end=self._parse_time(context.time_range["end"]),
                step=step,
            )
        return await self._connector.execute(rendered_query.query_text)

    def _prometheus_step(self, time_range: dict[str, Any], constraints: dict[str, Any]) -> str:
        from datetime import datetime
        import math

        max_points = int(constraints.get("max_points", 240))
        start = self._parse_time(time_range["start"])
        end = self._parse_time(time_range["end"])
        total_seconds = max(1, int((end - start).total_seconds()))
        step_seconds = max(1, math.ceil(total_seconds / max_points))
        return f"{step_seconds}s"

    def _parse_time(self, value: str):
        from datetime import datetime

        return parse_time_to_utc(value)

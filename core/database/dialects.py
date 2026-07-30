"""Database dialect registry for text-to-query planning.

This module keeps database-specific query-language rules out of the top-level
ReAct prompt and out of generic sql_query orchestration.  New TSDB backends
should add/extend a dialect here instead of patching caller code.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
import math
from typing import Any

from core.time_range import normalize_time_value

from .contracts import QueryRequestContext, RenderedQuery
from .query_compiler import QueryCompiler
from .query_plan import DatabaseQueryPlan, QueryFilter, TimeRangePlan


@dataclass(frozen=True)
class DatabaseDialect:
    """Query-generation and validation behavior for one database dialect."""

    database_types: tuple[str, ...]
    query_language: str
    read_only_languages: tuple[str, ...]
    generation_rules: str
    schema_linking_rules: str = ""
    sql_family: bool = False

    def render_plan(self, *, context: QueryRequestContext, plan: DatabaseQueryPlan, config: dict[str, Any]) -> RenderedQuery:
        compiled = QueryCompiler().compile(plan, db_type=self.normalized_type(context.database_type), dialect=self.normalized_type(context.database_type))
        return RenderedQuery(
            query_text=compiled.query,
            query_language=compiled.language or self.query_language,
            warnings=list(compiled.warnings),
        )

    def validate_read_only(self, query: str, query_language: str | None) -> None:
        stripped = query.strip()
        if not stripped:
            raise ValueError("sql_query requires a non-empty query.")
        language = self.normalize_query_language(query_language, query)
        if language in {"sql", "sqlite", "postgresql", "timescaledb", "questdb", "clickhouse"} or self.sql_family:
            if not re.match(r"^\s*(with|select)\b", stripped, re.IGNORECASE):
                raise ValueError("Only read-only SELECT/ WITH SQL analysis queries are allowed.")
            if _SQL_WRITE_PATTERN.search(stripped):
                raise ValueError("Write or DDL statements are not allowed in sql_query.")
            return
        if _SQL_WRITE_PATTERN.search(stripped):
            raise ValueError("Write or DDL statements are not allowed in sql_query.")

    def query_shape_issues(self, *, query: str, query_language: str | None = None, query_task_contract: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return []

    def repair_query(self, *, query: str, query_language: str | None = None, error: Exception | str | None = None):
        from .repair import QueryRepairResult

        return QueryRepairResult(query=query)

    def normalize_query_language(self, query_language: str | None, query: str | None = None) -> str:
        language = str(query_language or "").strip().lower()
        if language:
            return language
        return self.query_language

    def has_rendered_filter(self, rendered_query: RenderedQuery, item: QueryFilter) -> bool:
        text = rendered_query.query_text.lower()
        column = str(item.column).lower()
        value = str(item.value).lower()
        return column in text and value in text

    def internal_columns(self) -> set[str]:
        return {"time", "timestamp", "result", "table"}

    def projected_columns(self, *, query: str | None, query_language: str | None = None) -> set[str]:
        return set()

    def field_present_in_columns(self, field: str, columns: set[str]) -> bool:
        normalized_field = field.replace("-", "_").strip("_").lower()
        normalized_columns = {column.replace("-", "_").strip("_").lower() for column in columns}
        if normalized_field in normalized_columns:
            return True
        if normalized_field in {"time", "timestamp", "datetime", "date"} and normalized_columns & {"time", "timestamp", "datetime", "date"}:
            return True
        for column in normalized_columns:
            tokens = [token for token in column.split("_") if token]
            if normalized_field in tokens or column.endswith(f"_{normalized_field}"):
                return True
        return False

    def has_value_alias(self, columns: set[str]) -> bool:
        return bool(columns & {"value", "metric_value"})

    def raw_limit_without_downsampling(self, query: str | None, query_language: str | None = None) -> bool:
        return bool(re.search(r"\blimit\s+\d+\b", str(query or ""), flags=re.IGNORECASE))

    def markdown_language(self, query_language: str | None = None) -> str | None:
        normalized = str(query_language or self.query_language or "").strip().lower()
        if not normalized:
            return None
        if normalized in {"postgres", "postgresql", "timescaledb", "questdb", "clickhouse"}:
            return "sql"
        return normalized

    def supports_range_query(self, *, context: QueryRequestContext, intent_query_shape: str, connector: Any) -> bool:
        return False

    def range_step(self, *, time_range: dict[str, Any], constraints: dict[str, Any], parse_time) -> str | None:
        return None

    @staticmethod
    def normalized_type(database_type: str | None) -> str:
        return str(database_type or "").strip().lower()


class InfluxDBFluxDialect(DatabaseDialect):
    def __init__(self):
        super().__init__(
            database_types=("influxdb", "flux"),
            query_language="flux",
            read_only_languages=("flux",),
            generation_rules=(
                "For InfluxDB, generate Flux. Do not use to(), experimental.to(), or write functions. "
                "Influx fields are stored as _field/_value in normal long-form Flux results: select a field with "
                "r[\"_field\"] == \"field_name\" and aggregate/read the default _value column. "
                "Do not use min(column:\"field_name\"), max(column:\"field_name\"), keep(columns:[\"field_name\"]), "
                "or r[\"field_name\"] unless schema/sample explicitly proves a pivoted physical column. "
                "For raw time-series evidence, return _time, _value, _field, and relevant tag columns. "
                "For multiple aggregates, derived differences/ratios, or downstream code_interpreter work, prefer a simple "
                "raw filtered series and mark derived values missing rather than writing fragile Flux state logic."
            ),
            schema_linking_rules=(
                "For InfluxDB, treat measurements as sources, _field values as value columns, and tags as dimensions. "
                "User entities such as units, symbols, tickers, asset names, regions, or codes should normally become tag filters "
                "when candidate tag values support them."
            ),
        )

    def render_plan(self, *, context: QueryRequestContext, plan: DatabaseQueryPlan, config: dict[str, Any]) -> RenderedQuery:
        bucket = str(config.get("bucket") or config.get("database") or "")
        source = plan.sources[0] if plan.sources else None
        measurement = source.name if source else ""
        field_projection = next((item for item in plan.projections if item.alias == "value"), None)
        field_name = field_projection.column if field_projection else (source.value_columns[0] if source and source.value_columns else "_value")
        lines = [f'from(bucket: "{_escape_flux_string(bucket)}")', f"  |> {self._flux_range(plan.time_range, config)}"]
        if measurement:
            lines.append(f'  |> filter(fn: (r) => r["_measurement"] == "{_escape_flux_string(measurement)}")')
        if field_name:
            lines.append(f'  |> filter(fn: (r) => r["_field"] == "{_escape_flux_string(field_name)}")')
        for item in plan.filters:
            if item.column in {"_measurement", "_field", "_time", "time", "timestamp"}:
                continue
            lines.append(
                f'  |> filter(fn: (r) => r["{_escape_flux_string(item.column)}"] {self._flux_operator(item.operator)} "{_escape_flux_string(item.value)}")'
            )
        aggregation = next((item.aggregation for item in plan.projections if item.aggregation), None)
        if aggregation:
            flux_fn = {"avg": "mean", "mean": "mean", "max": "max", "min": "min", "sum": "sum", "count": "count"}.get(aggregation, aggregation)
            lines.append("  |> group()")
            lines.append(f"  |> {flux_fn}()")
        return RenderedQuery(query_text="\n".join(lines), query_language="flux")

    def validate_read_only(self, query: str, query_language: str | None) -> None:
        stripped = query.strip()
        if not stripped:
            raise ValueError("sql_query requires a non-empty query.")
        language = self.normalize_query_language(query_language, query)
        if language == "flux" or "|>" in stripped or stripped.startswith("from("):
            if _FLUX_WRITE_PATTERN.search(stripped):
                raise ValueError("Flux output/write functions are not allowed in sql_query.")
            return
        super().validate_read_only(query, query_language)

    def query_shape_issues(self, *, query: str, query_language: str | None = None, query_task_contract: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        normalized = str(query or "").lower()
        issues: list[dict[str, Any]] = []
        if re.search(r"\bsummarize\s*\(", normalized):
            issues.append(
                {
                    "code": "unsupported_flux_helper",
                    "message": "Flux query uses unsupported summarize() helper.",
                    "recommended_shape": "raw_series_or_simple_aggregate_table",
                }
            )
        if re.search(r"\{[^}]*\[[\"'][^\"']+[\"']\]\s*:", query or ""):
            issues.append(
                {
                    "code": "invalid_flux_record_property",
                    "message": "Flux record property keys cannot be dynamic bracket expressions in object literals.",
                    "recommended_shape": "raw_series_or_simple_aggregate_table",
                }
            )
        contract = query_task_contract if isinstance(query_task_contract, dict) else {}
        downstream = str(contract.get("downstream_action") or "").strip().lower()
        preferred = str(contract.get("preferred_evidence_shape") or "").strip().lower()
        if downstream == "code_interpreter" and preferred == "raw_series":
            aggregate_tokens = (
                "|> max(",
                "|> min(",
                "|> mean(",
                "|> sum(",
                "|> count(",
                "|> aggregatewindow(",
                "|> group(",
                "|> pivot(",
                "|> join(",
            )
            if any(token in normalized for token in aggregate_tokens):
                issues.append(
                    {
                        "code": "aggregate_used_for_downstream_raw_series",
                        "message": "Query contract asks for raw evidence for downstream code_interpreter but query aggregates the series.",
                        "recommended_shape": "raw_series",
                    }
                )
            keep_match = re.search(r"\|\>\s*keep\s*\(\s*columns\s*:\s*\[([^\]]*)\]", query or "", flags=re.IGNORECASE)
            if keep_match:
                kept = {item[0] or item[1] for item in re.findall(r'"([^"]+)"|\'([^\']+)\'', keep_match.group(1))}
                if "_value" not in kept or "_time" not in kept:
                    issues.append(
                        {
                            "code": "raw_series_missing_native_time_value",
                            "message": "Raw Flux evidence must retain native _time and _value columns.",
                            "recommended_shape": "raw_series",
                        }
                    )
        if re.search(r"\|\>\s*aggregatewindow\s*\(", normalized):
            keep_match = re.search(r"\|\>\s*keep\s*\(\s*columns\s*:\s*\[([^\]]*)\]", query or "", flags=re.IGNORECASE)
            if keep_match and "_value" not in keep_match.group(1):
                issues.append(
                    {
                        "code": "flux_aggregate_missing_value_column",
                        "message": "aggregateWindow operates on _value by default, but the preceding keep() drops _value.",
                        "recommended_shape": "Keep _value for long-form Flux aggregates, or filter _field and return raw _time/_value evidence.",
                    }
                )
        return issues

    def repair_query(self, *, query: str, query_language: str | None = None, error: Exception | str | None = None):
        from .repair import QueryRepairResult

        error_text = str(error or "").lower()
        raw_keep_result = self._repair_raw_keep_projection(query, error_text)
        if raw_keep_result.changed:
            return raw_keep_result
        date_result = self._repair_date_import(query, error_text)
        if date_result.changed:
            return date_result
        yield_result = self._repair_duplicate_default_result(query, error_text)
        if yield_result.changed:
            return yield_result
        return QueryRepairResult(query=query)

    def _repair_raw_keep_projection(self, query: str, error_text: str):
        from .repair import QueryRepairResult

        if "raw_series_missing_native_time_value" not in error_text and "raw flux evidence" not in error_text:
            return QueryRepairResult(query=query)

        def replace_keep(match: re.Match) -> str:
            raw_columns = match.group(1)
            columns = [item[0] or item[1] for item in re.findall(r'"([^"]+)"|\'([^\']+)\'', raw_columns)]
            preserved_tags = [
                column
                for column in columns
                if column not in {"_time", "_value", "_field", "time", "timestamp", "value", "price"}
                and not column.startswith("_")
            ]
            repaired = ["_time", "_value", "_field", *preserved_tags]
            rendered = ", ".join(f'"{column}"' for column in dict.fromkeys(repaired))
            return f"keep(columns: [{rendered}]"

        repaired = re.sub(
            r"keep\s*\(\s*columns\s*:\s*\[([^\]]*)\]",
            replace_keep,
            query,
            flags=re.IGNORECASE,
            count=1,
        )
        if repaired != query:
            return QueryRepairResult(query=repaired, changed=True, reason="repaired_flux_raw_keep_projection")
        return QueryRepairResult(query=query)

    def normalize_query_language(self, query_language: str | None, query: str | None = None) -> str:
        language = str(query_language or "").strip().lower()
        if language:
            return language
        query_text = str(query or "").strip()
        if "|>" in query_text or query_text.startswith("from("):
            return "flux"
        return "flux"

    def has_rendered_filter(self, rendered_query: RenderedQuery, item: QueryFilter) -> bool:
        text = rendered_query.query_text.lower()
        column = str(item.column).lower()
        value = str(item.value).lower()
        column_patterns = [f"r.{column}", f'r["{column}"]', f"r['{column}']"]
        value_patterns = [f'"{value}"', f"'{value}'"]
        return any(pattern in text for pattern in column_patterns) and any(pattern in text for pattern in value_patterns)

    def internal_columns(self) -> set[str]:
        return {"_measurement", "_field", "_time", "_start", "_stop", "time", "timestamp", "result", "table"}

    def projected_columns(self, *, query: str | None, query_language: str | None = None) -> set[str]:
        if not query:
            return set()
        projected: set[str] = set()
        for match in re.finditer(r"keep\s*\(\s*columns\s*:\s*\[([^\]]*)\]", query, flags=re.IGNORECASE | re.DOTALL):
            for item in re.findall(r'"([^"]+)"|\'([^\']+)\'', match.group(1)):
                column = (item[0] or item[1]).strip().lower()
                if column:
                    projected.add(column)
        return _apply_renames(projected, query)

    def has_value_alias(self, columns: set[str]) -> bool:
        return bool(columns & {"value", "_value", "metric_value"})

    def raw_limit_without_downsampling(self, query: str | None, query_language: str | None = None) -> bool:
        normalized_query = str(query or "").lower()
        return "limit(" in normalized_query and "aggregatewindow" not in normalized_query

    def markdown_language(self, query_language: str | None = None) -> str | None:
        return "flux"

    def _flux_range(self, time_range: TimeRangePlan, config: dict[str, Any]) -> str:
        if time_range.start and time_range.end:
            return f'range(start: {normalize_time_value(time_range.start)}, stop: {normalize_time_value(time_range.end)})'
        if time_range.start:
            return f'range(start: {normalize_time_value(time_range.start)})'
        if time_range.lookback:
            return f"range(start: -{time_range.lookback})"
        default_range = _default_time_range(config)
        if default_range.get("start") and default_range.get("end"):
            return f'range(start: {normalize_time_value(default_range["start"])}, stop: {normalize_time_value(default_range["end"])})'
        if default_range.get("start"):
            return f'range(start: {normalize_time_value(default_range["start"])})'
        return "range(start: 1970-01-01T00:00:00Z)"

    @staticmethod
    def _flux_operator(operator: str) -> str:
        return "==" if operator == "=" else operator

    @staticmethod
    def _repair_date_import(query: str, error_text: str):
        from .repair import QueryRepairResult

        if "date." not in query:
            return QueryRepairResult(query=query)
        if 'import "date"' in query or "import 'date'" in query:
            return QueryRepairResult(query=query)
        if error_text and "undefined identifier date" not in error_text:
            return QueryRepairResult(query=query)
        return QueryRepairResult(
            query='import "date"\n' + query,
            changed=True,
            reason="add_flux_date_import",
            hint='Added Flux import "date" because the query uses date.* functions.',
        )

    @staticmethod
    def _repair_duplicate_default_result(query: str, error_text: str):
        from .repair import QueryRepairResult

        if "tried to produce more than one result" not in error_text:
            return QueryRepairResult(query=query)
        if "yield(" in query:
            return QueryRepairResult(
                query=query,
                hint="Flux produced multiple default results; split the query or give each result a unique yield(name).",
            )
        parts = [part.strip() for part in re.split(r"\n\s*\n(?=from\s*\()", query) if part.strip()]
        if len(parts) < 2:
            return QueryRepairResult(
                query=query,
                hint="Flux produced multiple default results; split the query or give each result a unique yield(name).",
            )
        repaired_parts = [
            f'{part}\n  |> yield(name: "result_{index}")'
            for index, part in enumerate(parts, start=1)
        ]
        return QueryRepairResult(
            query="\n\n".join(repaired_parts),
            changed=True,
            reason="name_flux_results",
            hint="Added unique yield names to multiple Flux result streams.",
        )


class PrometheusDialect(DatabaseDialect):
    def __init__(self):
        super().__init__(
            database_types=("prometheus", "promql"),
            query_language="promql",
            read_only_languages=("promql",),
            generation_rules=(
                "For Prometheus, generate PromQL. Treat metric names as sources and labels as dimensions. "
                "Preserve user-mentioned labels as matchers when grounded by schema or candidate values."
            ),
            schema_linking_rules="For Prometheus, treat metric names as sources and labels as dimensions.",
        )

    def render_plan(self, *, context: QueryRequestContext, plan: DatabaseQueryPlan, config: dict[str, Any]) -> RenderedQuery:
        source = plan.sources[0] if plan.sources else None
        metric_name = source.name if source else ""
        label_filters = [item for item in plan.filters if item.column not in {"timestamp", "time", "_time"}]
        matcher_text = ""
        if label_filters:
            matcher_text = "{" + ",".join(f'{item.column}="{item.value}"' for item in label_filters) + "}"
        return RenderedQuery(query_text=f"{metric_name}{matcher_text}".strip(), query_language="promql")

    def has_rendered_filter(self, rendered_query: RenderedQuery, item: QueryFilter) -> bool:
        text = rendered_query.query_text.lower()
        column = str(item.column).lower()
        value = str(item.value).lower()
        return column in text and (f'"{value}"' in text or f"'{value}'" in text)

    def markdown_language(self, query_language: str | None = None) -> str | None:
        return "promql"

    def supports_range_query(self, *, context: QueryRequestContext, intent_query_shape: str, connector: Any) -> bool:
        return bool(context.time_range and intent_query_shape == "raw_timeseries" and hasattr(connector, "get_range"))

    def range_step(self, *, time_range: dict[str, Any], constraints: dict[str, Any], parse_time) -> str | None:
        max_points = int(constraints.get("max_points", 240))
        start = parse_time(time_range["start"])
        end = parse_time(time_range["end"])
        total_seconds = max(1, int((end - start).total_seconds()))
        step_seconds = max(1, math.ceil(total_seconds / max_points))
        return f"{step_seconds}s"


class SqlFamilyDialect(DatabaseDialect):
    def __init__(self, database_types: tuple[str, ...] = ("sql", "postgresql", "timescaledb", "questdb", "clickhouse")):
        super().__init__(
            database_types=database_types,
            query_language="sql",
            read_only_languages=database_types,
            sql_family=True,
            generation_rules=(
                "For SQL-family databases, generate read-only SELECT or WITH queries only. "
                "Use the physical table, timestamp column, value columns, and dimensions from schema_preview/schema_linking. "
                "Preserve required filters, grouping, ordering, and time predicates explicitly in the query."
            ),
            schema_linking_rules=(
                "For SQL-family databases, treat tables/views as sources, normal columns as value/time/dimension columns, "
                "and user entities as equality filters when grounded by candidate values."
            ),
        )


_SQL_WRITE_PATTERN = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|create|replace|merge|grant|revoke|vacuum|attach|detach|pragma)\b",
    re.IGNORECASE,
)
_FLUX_WRITE_PATTERN = re.compile(r"\bto\s*\(|experimental\.to\s*\(", re.IGNORECASE)


_DIALECTS: tuple[DatabaseDialect, ...] = (
    InfluxDBFluxDialect(),
    PrometheusDialect(),
    SqlFamilyDialect(),
)


def dialect_for_database(database_type: str | None) -> DatabaseDialect:
    normalized = DatabaseDialect.normalized_type(database_type)
    for dialect in _DIALECTS:
        if normalized in dialect.database_types:
            return dialect
    return SqlFamilyDialect(database_types=(normalized or "sql",))


def query_language_for_database_type(database_type: str | None) -> str:
    return dialect_for_database(database_type).query_language


def _default_time_range(config: dict[str, Any]) -> dict[str, Any]:
    configured = config.get("default_query_time_range") or config.get("default_time_range")
    if isinstance(configured, dict):
        return _normalize_time_range(configured)
    reference_dataset = config.get("reference_dataset")
    if isinstance(reference_dataset, dict):
        configured = reference_dataset.get("time_range")
        if isinstance(configured, dict):
            return _normalize_time_range(configured)
    return {"start": "1970-01-01T00:00:00Z"}


def _normalize_time_range(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: raw
        for key, raw in {
            "start": value.get("start") or value.get("from"),
            "end": value.get("end") or value.get("stop") or value.get("to"),
            "lookback": value.get("lookback"),
        }.items()
        if raw not in (None, "")
    }


def _escape_flux_string(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _apply_renames(projected: set[str], query: str) -> set[str]:
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

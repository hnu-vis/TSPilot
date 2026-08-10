"""Database dialect registry for text-to-query planning.

This module keeps database-specific query-language rules out of the top-level
ReAct prompt and out of generic sql_query orchestration.  New TSDB backends
should add/extend a dialect here instead of patching caller code.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DatabaseDialect:
    """Query-generation and validation behavior for one database dialect."""

    database_types: tuple[str, ...]
    query_language: str
    read_only_languages: tuple[str, ...]
    generation_rules: str
    schema_linking_rules: str = ""
    sql_family: bool = False

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

    def schema_preview_extensions(self, *, schema: Any, preview: dict[str, Any]) -> dict[str, Any]:
        return {}

    def normalize_query_language(self, query_language: str | None, query: str | None = None) -> str:
        language = str(query_language or "").strip().lower()
        if language:
            return language
        return self.query_language

    def has_filter(self, query: str, *, column: str, value: Any) -> bool:
        text = query.lower()
        column = str(column).lower()
        value = str(value).lower()
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

    def schema_preview_extensions(self, *, schema: Any, preview: dict[str, Any]) -> dict[str, Any]:
        value_domains = getattr(schema, "metadata", {}).get("value_domains")
        if not isinstance(value_domains, dict):
            return {}
        tables = preview.get("tables_or_measurements") if isinstance(preview.get("tables_or_measurements"), list) else []
        measure_mappings = []
        for table in tables:
            if not isinstance(table, dict):
                continue
            table_name = table.get("name")
            field_values = table.get("field_values") if isinstance(table.get("field_values"), list) else []
            if field_values:
                table["field_value_semantics"] = {
                    "logical_measure_source": "_field",
                    "physical_value_column": "_value",
                    "aggregate_column": "_value",
                    "rule": "Filter _field to select a logical measure; aggregate/read numeric values from _value.",
                }
            for field_value in field_values:
                if field_value in (None, ""):
                    continue
                measure_mappings.append(
                    {
                        "source": table_name,
                        "logical_measure": str(field_value),
                        "selector_column": "_field",
                        "selector_value": str(field_value),
                        "physical_value_column": "_value",
                        "aggregate_column": "_value",
                        "time_column": "_time",
                    }
                )
        if not measure_mappings:
            return {}
        return {
            "physical_model": {
                "dialect": "influxdb_flux_long_form",
                "value_storage": "logical measures are selected via _field; numeric samples are stored in _value",
                "measure_mappings": measure_mappings[:200],
                "query_generation_constraints": [
                    "Filter r[\"_field\"] to select logical_measure.",
                    "Use _value for aggregate functions such as max/min/mean/sum/count unless a pivoted physical column is explicitly proven.",
                    "Do not aggregate field values directly when the field is listed as logical_measure.",
                ],
            }
        }

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
        issues.extend(self._logical_field_aggregate_issues(query=query, query_task_contract=contract))
        return issues

    def _logical_field_aggregate_issues(self, *, query: str | None, query_task_contract: dict[str, Any]) -> list[dict[str, Any]]:
        text = str(query or "")
        if not text.strip():
            return []
        mappings = self._measure_mappings_from_contract(query_task_contract)
        if not mappings:
            return []
        issues: list[dict[str, Any]] = []
        aggregate_pattern = re.compile(
            r"\|\>\s*(?P<fn>max|min|mean|sum|count)\s*\(\s*column\s*:\s*[\"'](?P<column>[^\"']+)[\"']",
            flags=re.IGNORECASE,
        )
        for match in aggregate_pattern.finditer(text):
            physical_column = match.group("column")
            mapping = next(
                (
                    item for item in mappings
                    if str(item.get("logical_measure") or item.get("selector_value") or "").lower() == physical_column.lower()
                    and str(item.get("physical_value_column") or item.get("aggregate_column") or "").lower() != physical_column.lower()
                ),
                None,
            )
            if not mapping:
                continue
            issues.append(
                {
                    "code": "flux_logical_field_used_as_physical_aggregate_column",
                    "message": (
                        f"Flux query aggregates logical field value {physical_column!r} as a physical column. "
                        "Long-form Influx data must filter the field selector and aggregate the physical value column."
                    ),
                    "failed_physical_column": physical_column,
                    "logical_measure": mapping.get("logical_measure") or mapping.get("selector_value"),
                    "required_field_filter": {
                        "column": mapping.get("selector_column") or "_field",
                        "operator": "=",
                        "value": mapping.get("selector_value") or mapping.get("logical_measure"),
                    },
                    "physical_value_column": mapping.get("physical_value_column") or "_value",
                    "aggregate_column": mapping.get("aggregate_column") or mapping.get("physical_value_column") or "_value",
                    "recommended_shape": "schema_grounded_flux_aggregate",
                }
            )
        return issues

    def _measure_mappings_from_contract(self, contract: dict[str, Any]) -> list[dict[str, Any]]:
        candidates: list[Any] = []
        for key in ("measures", "aggregate_targets"):
            value = contract.get(key)
            if isinstance(value, list):
                candidates.extend(value)
        diagnostics = contract.get("_schema_linking_diagnostics")
        if isinstance(diagnostics, dict):
            for key in ("measure_mappings", "measures", "aggregate_targets"):
                value = diagnostics.get(key)
                if isinstance(value, list):
                    candidates.extend(value)
        mappings = []
        seen = set()
        for item in candidates:
            if not isinstance(item, dict):
                continue
            logical = item.get("logical_measure") or item.get("selector_value")
            physical = item.get("physical_value_column") or item.get("aggregate_column")
            if not logical or not physical:
                continue
            key = (str(logical).lower(), str(physical).lower())
            if key in seen:
                continue
            seen.add(key)
            mappings.append(item)
        return mappings

    def normalize_query_language(self, query_language: str | None, query: str | None = None) -> str:
        language = str(query_language or "").strip().lower()
        if language:
            return language
        query_text = str(query or "").strip()
        if "|>" in query_text or query_text.startswith("from("):
            return "flux"
        return "flux"

    def has_filter(self, query: str, *, column: str, value: Any) -> bool:
        text = query.lower()
        column = str(column).lower()
        raw_value = value
        normalized_value = str(value).lower()
        column_patterns = [f"r.{column}", f'r["{column}"]', f"r['{column}']"]
        value_patterns = [f'"{normalized_value}"', f"'{normalized_value}'"]
        if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
            value_patterns.append(rf"(?<![\w.]){re.escape(normalized_value)}(?![\w.])")
        has_column = any(pattern in text for pattern in column_patterns)
        has_value = any(
            re.search(pattern, text) if pattern.startswith("(?<!") else pattern in text
            for pattern in value_patterns
        )
        return has_column and has_value

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

class PrometheusDialect(DatabaseDialect):
    def __init__(self):
        super().__init__(
            database_types=("prometheus", "promql", "victoriametrics", "m3db"),
            query_language="promql",
            read_only_languages=("promql",),
            generation_rules=(
                "For Prometheus, generate PromQL. Treat metric names as sources and labels as dimensions. "
                "Return one complete PromQL expression per query and never join independent expressions with newline or semicolon separators. "
                "A metric source is written directly as the PromQL selector name; never turn it into a source/name/__name__ label matcher. "
                "Only actual label columns may appear inside braces. Preserve user-mentioned labels as matchers when grounded by schema or candidate values. "
                "Use native PromQL functions such as rate(metric[window]) for rate requests; raw counter samples do not satisfy a rate request. "
                "For range results, request range evaluation rather than returning a bare range vector. "
                "When combining multiple metric sources whose label sets can overlap, preserve source identity before the or union with "
                "label_replace(metric, \"metric_name\", \"$1\", \"__name__\", \"(.*)\")."
            ),
            schema_linking_rules=(
                "For Prometheus, metric names are physical sources, never selector columns, selector values, dimensions, or required filters. "
                "Set physical_value_column and aggregate_column to value. Only columns listed on a metric source other than timestamp/value are labels. "
                "Rate windows and relative time ranges are temporal query semantics, not unresolved schema terms."
            ),
        )

    def has_filter(self, query: str, *, column: str, value: Any) -> bool:
        text = query.lower()
        column = str(column).lower()
        value = str(value).lower()
        return column in text and (f'"{value}"' in text or f"'{value}'" in text)

    def markdown_language(self, query_language: str | None = None) -> str | None:
        return "promql"


class SqlFamilyDialect(DatabaseDialect):
    def __init__(self, database_types: tuple[str, ...] = ("sql", "postgresql", "timescaledb", "questdb", "clickhouse", "openmldb", "greptimedb", "tdengine", "cnosdb", "arcadedb", "cratedb", "druid", "influxdb3", "griddb", "machbase", "nsdb", "axibase", "opengemini", "db2", "timestream", "riak_ts", "dolphindb", "kdb", "raimadb", "extremedb", "ittiadb", "irondb", "bangdb", "arc")):
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

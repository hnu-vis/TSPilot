"""Compile structured database query plans into physical query payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from .query_plan import DatabaseQueryPlan


@dataclass
class CompiledQuery:
    """A physical query generated from a structured query plan."""

    query: str
    language: str
    execution_strategy: str = "single_query"
    warnings: list[str] = field(default_factory=list)
    plan: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        """Return the query payload shape used by database agents."""
        payload = {
            "query": self.query,
            "language": self.language,
            "reasoning": "Compiled from structured database query plan.",
            "execution_strategy": self.execution_strategy,
        }
        if self.warnings:
            payload["warnings"] = self.warnings
        if self.plan:
            payload["query_plan"] = self.plan
        return payload


class QueryCompiler:
    """Dispatch query-plan compilation by database dialect."""

    SQL_DIALECTS = {"postgresql", "timescaledb", "questdb", "clickhouse", "openmldb", "greptimedb", "tdengine", "cnosdb", "arcadedb", "cratedb", "druid", "influxdb3", "griddb", "machbase", "nsdb", "axibase", "opengemini", "db2", "timestream", "riak_ts", "dolphindb", "kdb", "raimadb", "extremedb", "ittiadb", "irondb", "bangdb", "arc", "generic", "sql"}

    def compile(
        self,
        plan: DatabaseQueryPlan,
        *,
        db_type: str,
        dialect: str,
    ) -> CompiledQuery:
        """Compile a structured plan into a physical query payload."""
        normalized = (dialect or db_type or "sql").lower()
        if normalized in self.SQL_DIALECTS or db_type in self.SQL_DIALECTS:
            return SQLQueryCompiler().compile(plan, dialect=normalized)
        return CompiledQuery(
            query="",
            language=normalized or "unknown",
            execution_strategy=plan.execution_strategy,
            warnings=[
                (
                    f"Structured query-plan compiler for {normalized!r} is not implemented yet; "
                    "use existing query generation or multi-query execution."
                )
            ],
            plan=plan.to_dict(),
        )


class SQLQueryCompiler:
    """Compile query plans to conservative SQL."""

    def compile(self, plan: DatabaseQueryPlan, *, dialect: str = "sql") -> CompiledQuery:
        """Compile supported parts of a query plan into SQL."""
        if not plan.sources:
            return CompiledQuery(
                query="",
                language="sql",
                execution_strategy="unknown",
                warnings=["Cannot compile SQL query without linked sources."],
                plan=plan.to_dict(),
            )

        projections = self._compile_projections(plan, dialect=dialect)
        from_clause = self._compile_from(plan, dialect=dialect)
        where_clause = self._compile_filters(plan, dialect=dialect)
        sql_parts = [f"SELECT {projections}", from_clause]
        if where_clause:
            sql_parts.append(where_clause)
        query = "\n".join(sql_parts)
        warnings = []
        if len(plan.sources) > 1 and not plan.joins:
            warnings.append("Multiple sources were linked but no join was planned.")
        return CompiledQuery(
            query=query,
            language="sql",
            execution_strategy=plan.execution_strategy,
            warnings=warnings,
            plan=plan.to_dict(),
        )

    def _compile_projections(self, plan: DatabaseQueryPlan, *, dialect: str) -> str:
        if not plan.projections:
            return "*"
        parts = []
        seen = set()
        for projection in plan.projections:
            column = self._quote_identifier(projection.column, dialect=dialect)
            if projection.source:
                column = f"{self._quote_identifier(projection.source, dialect=dialect)}.{column}"
            expression = column
            if projection.aggregation:
                expression = f"{projection.aggregation.upper()}({column})"
            alias = projection.alias
            if alias:
                expression = f"{expression} AS {self._quote_identifier(alias, dialect=dialect)}"
            if expression not in seen:
                parts.append(expression)
                seen.add(expression)
        return ", ".join(parts) if parts else "*"

    def _compile_from(self, plan: DatabaseQueryPlan, *, dialect: str) -> str:
        first = plan.sources[0]
        clause = f"FROM {self._qualified_source(first.schema, first.name, dialect=dialect)}"
        if first.alias:
            clause += f" AS {self._quote_identifier(first.alias, dialect=dialect)}"
        for join in plan.joins:
            right = self._source_by_alias_or_name(plan, join.right)
            if not right:
                continue
            join_type = "JOIN" if join.type == "inner" else f"{join.type.upper()} JOIN"
            clause += f"\n{join_type} {self._qualified_source(right.schema, right.name, dialect=dialect)}"
            if right.alias:
                clause += f" AS {self._quote_identifier(right.alias, dialect=dialect)}"
            conditions = []
            for key in join.keys:
                left_key = self._quote_identifier(key, dialect=dialect)
                right_key = self._quote_identifier(key, dialect=dialect)
                conditions.append(
                    f"{self._quote_identifier(join.left, dialect=dialect)}.{left_key} = "
                    f"{self._quote_identifier(join.right, dialect=dialect)}.{right_key}"
                )
            if not conditions and plan.alignment.time_column:
                time_key = self._quote_identifier(plan.alignment.time_column, dialect=dialect)
                conditions.append(
                    f"{self._quote_identifier(join.left, dialect=dialect)}.{time_key} = "
                    f"{self._quote_identifier(join.right, dialect=dialect)}.{time_key}"
                )
            clause += " ON " + (" AND ".join(conditions) if conditions else "1 = 1")
        return clause

    def _compile_filters(self, plan: DatabaseQueryPlan, *, dialect: str) -> str:
        filters = []
        for item in plan.filters:
            column = self._quote_identifier(item.column, dialect=dialect)
            if item.source:
                column = f"{self._quote_identifier(item.source, dialect=dialect)}.{column}"
            filters.append(f"{column} {item.operator} {self._literal(item.value)}")
        time_filter = self._compile_time_range_filter(plan, dialect=dialect)
        if time_filter:
            filters.append(time_filter)
        return "WHERE " + " AND ".join(filters) if filters else ""

    def _compile_time_range_filter(self, plan: DatabaseQueryPlan, *, dialect: str) -> str | None:
        if not plan.alignment.time_column or not plan.sources:
            return None
        source = plan.sources[0].alias or plan.sources[0].name
        time_column = (
            f"{self._quote_identifier(source, dialect=dialect)}."
            f"{self._quote_identifier(plan.alignment.time_column, dialect=dialect)}"
        )
        if plan.time_range.start and plan.time_range.end:
            return (
                f"{time_column} >= {self._literal(plan.time_range.start)} "
                f"AND {time_column} <= {self._literal(plan.time_range.end)}"
            )
        if plan.time_range.start:
            return f"{time_column} >= {self._literal(plan.time_range.start)}"
        if plan.time_range.end:
            return f"{time_column} <= {self._literal(plan.time_range.end)}"
        if plan.time_range.lookback:
            return f"{time_column} >= {self._lookback_expression(plan.time_range.lookback, dialect=dialect)}"
        return None

    def _source_by_alias_or_name(self, plan: DatabaseQueryPlan, value: str):
        for source in plan.sources:
            if value in {source.alias, source.name}:
                return source
        return None

    def _qualified_source(self, schema: str, name: str, *, dialect: str) -> str:
        if schema:
            return f"{self._quote_identifier(schema, dialect=dialect)}.{self._quote_identifier(name, dialect=dialect)}"
        return self._quote_identifier(name, dialect=dialect)

    def _quote_identifier(self, value: str, *, dialect: str) -> str:
        if dialect == "clickhouse":
            escaped = str(value).replace("`", "``")
            return f"`{escaped}`"
        escaped = str(value).replace('"', '""')
        return f'"{escaped}"'

    def _lookback_expression(self, lookback: str, *, dialect: str) -> str:
        interval = self._format_interval(lookback, dialect=dialect)
        if dialect == "clickhouse":
            return f"now() - {interval}"
        return f"NOW() - {interval}"

    def _format_interval(self, lookback: str, *, dialect: str) -> str:
        text = str(lookback or "").strip()
        match = re.fullmatch(r"(\d+)\s*([smhdw])", text, flags=re.IGNORECASE)
        if match:
            amount = int(match.group(1))
            unit = match.group(2).lower()
            clickhouse_units = {
                "s": "SECOND",
                "m": "MINUTE",
                "h": "HOUR",
                "d": "DAY",
                "w": "WEEK",
            }
            postgres_units = {
                "s": "seconds",
                "m": "minutes",
                "h": "hours",
                "d": "days",
                "w": "weeks",
            }
            if dialect == "clickhouse":
                return f"INTERVAL {amount} {clickhouse_units[unit]}"
            return f"INTERVAL '{amount} {postgres_units[unit]}'"
        if dialect == "clickhouse":
            return f"INTERVAL {self._literal(text)} SECOND"
        return f"INTERVAL {self._literal(text)}"

    def _literal(self, value: Any) -> str:
        if value is None:
            return "NULL"
        if isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        if isinstance(value, (int, float)):
            return str(value)
        return "'" + str(value).replace("'", "''") + "'"

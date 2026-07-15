"""Query translator for natural language to SQL conversion."""
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .connector import DatabaseSchema


@dataclass
class TranslationResult:
    """Result of query translation."""
    success: bool
    sql: str | None = None
    dialect: str = ""
    explanation: str | None = None
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result of query validation."""
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


class QueryTranslator:
    """Translates natural language queries to SQL.

    Uses LLM for complex translation with fallback to rule-based patterns.
    """

    TRANSLATION_PROMPT = """You are a SQL expert for time-series databases.

Translate the following natural language query to SQL for the specified database schema.

Database: {database}
Dialect: {dialect}

Schema:
{schema}

Query: {query}

Return a JSON object with:
{{"sql": "SELECT ...", "explanation": "brief explanation", "confidence": 0.0-1.0}}

Return ONLY the JSON object, nothing else."""

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        dialect_handlers: dict[str, Any] | None = None,
    ):
        self._llm = llm
        self._dialect_handlers = dialect_handlers or {}

    async def translate_nl_to_sql(
        self,
        natural_language: str,
        database: str,
        schema: DatabaseSchema,
    ) -> TranslationResult:
        """Translate natural language to SQL."""
        if self._llm:
            return await self._translate_with_llm(natural_language, database, schema)
        return self._translate_with_rules(natural_language, database, schema)

    async def _translate_with_llm(
        self,
        natural_language: str,
        database: str,
        schema: DatabaseSchema,
    ) -> TranslationResult:
        """Translate using LLM."""
        schema_str = self._format_schema(schema)

        prompt = self.TRANSLATION_PROMPT.format(
            database=database,
            dialect=self._get_dialect(database),
            schema=schema_str,
            query=natural_language,
        )

        try:
            result = await self._llm.ainvoke([
                SystemMessage(content="You are a SQL expert."),
                HumanMessage(content=prompt)
            ])

            content = result.content if hasattr(result, 'content') else str(result)
            parsed = self._parse_json_response(content)

            if parsed:
                return TranslationResult(
                    success=True,
                    sql=parsed.get("sql"),
                    dialect=self._get_dialect(database),
                    explanation=parsed.get("explanation"),
                    confidence=parsed.get("confidence", 0.8),
                )

            return TranslationResult(
                success=False,
                errors=["Failed to parse LLM response"],
                dialect=self._get_dialect(database),
            )

        except Exception as e:
            return TranslationResult(
                success=False,
                errors=[str(e)],
                dialect=self._get_dialect(database),
            )

    def _translate_with_rules(
        self,
        natural_language: str,
        database: str,
        schema: DatabaseSchema,
    ) -> TranslationResult:
        """Translate using rule-based patterns."""
        dialect = self._get_dialect(database)
        query_lower = natural_language.lower()

        # Simple rule-based translation for common patterns
        sql = self._apply_rules(query_lower, schema, dialect)

        return TranslationResult(
            success=True,
            sql=sql,
            dialect=dialect,
            explanation="Rule-based translation",
            confidence=0.6,
            warnings=["LLM not available, using limited rule-based translation"],
        )

    def _apply_rules(
        self,
        query: str,
        schema: DatabaseSchema,
        dialect: str,
    ) -> str:
        """Apply rule-based translation patterns."""
        sql_parts = []

        # SELECT
        if "select" in query or "get" in query or "show" in query:
            # Check for aggregation keywords
            if "max" in query or "highest" in query or "top" in query:
                agg = "MAX"
            elif "min" in query or "lowest" in query:
                agg = "MIN"
            elif "avg" in query or "average" in query or "mean" in query:
                agg = "AVG"
            elif "count" in query:
                agg = "COUNT"
            elif "sum" in query or "total" in query:
                agg = "SUM"
            else:
                agg = "*"
            sql_parts.append(f"SELECT {agg}")

            # Extract table name
            for table in schema.tables:
                if table.name in query:
                    sql_parts.append(f"FROM {table.name}")
                    break
            else:
                sql_parts.append("FROM <table>")

            # WHERE clause
            where_parts = []
            if "where" in query:
                if "time >" in query or "after" in query:
                    if dialect == "influxdb":
                        where_parts.append("time > now() - 1h")
                    elif dialect == "prometheus":
                        where_parts.append("__time__ > timestamp() - 3600")
                    else:
                        where_parts.append("timestamp > NOW() - INTERVAL '1 hour'")

                # Value filters
                if "above" in query or "over" in query:
                    num_match = re.search(r'\d+', query)
                    if num_match:
                        where_parts.append(f"value > {num_match.group()}")

            if where_parts:
                sql_parts.append("WHERE " + " AND ".join(where_parts))

            # GROUP BY
            if "group by" in query or "by host" in query or "by machine" in query:
                if "host" in query:
                    sql_parts.append("GROUP BY host")
                    sql_parts.append("ORDER BY MAX(value) DESC")

            # LIMIT
            limit_match = re.search(r'top\s+(\d+)|limit\s+(\d+)', query)
            if limit_match:
                limit_val = limit_match.group(1) or limit_match.group(2)
                sql_parts.append(f"LIMIT {limit_val}")

        else:
            sql_parts.append("SELECT * FROM <table>")

        return " ".join(sql_parts)

    def _format_schema(self, schema: DatabaseSchema) -> str:
        """Format schema for prompt inclusion."""
        lines = []
        for table in schema.tables:
            cols = ", ".join([f"{c.name} ({c.data_type})" for c in table.columns])
            lines.append(f"- {table.name}: {cols}")
        return "\n".join(lines) if lines else "No schema available"

    def _get_dialect(self, database: str) -> str:
        """Get SQL dialect for database type."""
        if "influx" in database.lower():
            return "influxdb"
        elif "timescale" in database.lower() or "postgres" in database.lower():
            return "postgresql"
        elif "prometheus" in database.lower():
            return "promql"
        elif "iotdb" in database.lower():
            return "iotdb"
        elif "questdb" in database.lower():
            return "questdb"
        elif "clickhouse" in database.lower():
            return "clickhouse"
        return "generic"

    def _parse_json_response(self, text: str) -> dict | None:
        """Parse JSON from LLM response."""
        text = text.strip()
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                import json
                return json.loads(json_match.group())
            except Exception:
                pass
        return None

    async def translate_sql_dialect(
        self,
        sql: str,
        from_dialect: str,
        to_dialect: str,
    ) -> TranslationResult:
        """Translate SQL between different dialects."""
        if from_dialect == to_dialect:
            return TranslationResult(
                success=True,
                sql=sql,
                dialect=to_dialect,
                confidence=1.0,
            )

        # Simple dialect translation rules
        translated_sql = sql

        # InfluxDB to PostgreSQL
        if from_dialect == "influxdb" and to_dialect == "postgresql":
            translated_sql = sql.replace("time > now()", "timestamp > NOW()")
            translated_sql = translated_sql.replace("time > now() - ", "timestamp > NOW() - INTERVAL '")
            if "INTERVAL" in translated_sql and not translated_sql.endswith("'"):
                translated_sql = translated_sql.replace("INTERVAL '", "INTERVAL '") + "'"
            translated_sql = translated_sql.replace("GROUP BY time(", "GROUP BY DATE_TRUNC(")
            translated_sql = translated_sql.replace("fill(previous)", "IGNORE NULLS")

        # PostgreSQL to InfluxDB
        elif from_dialect == "postgresql" and to_dialect == "influxdb":
            translated_sql = sql.replace("timestamp > NOW() - INTERVAL '", "time > now() - ")
            translated_sql = translated_sql.replace("DATE_TRUNC(", "time(")
            translated_sql = translated_sql.replace("IGNORE NULLS", "fill(previous)")

        return TranslationResult(
            success=True,
            sql=translated_sql,
            dialect=to_dialect,
            explanation=f"Translated from {from_dialect} to {to_dialect}",
            confidence=0.85,
        )

    async def validate_query(
        self,
        sql: str,
        dialect: str,
        schema: DatabaseSchema | None = None,
    ) -> ValidationResult:
        """Validate SQL query syntax and semantics."""
        errors = []
        warnings = []

        # Basic syntax validation
        sql_upper = sql.upper()

        if not sql_upper.startswith(("SELECT", "WITH", "SHOW", "EXPLAIN")):
            errors.append("Query must start with SELECT, WITH, SHOW, or EXPLAIN")

        # Check for unbalanced parentheses
        if sql.count("(") != sql.count(")"):
            errors.append("Unbalanced parentheses")

        # Check for SQL injection patterns
        dangerous_patterns = ["--", "DROP", "DELETE", "TRUNCATE", "ALTER", "INSERT", "UPDATE"]
        for pattern in dangerous_patterns:
            if pattern in sql_upper and not sql_upper.startswith(pattern):
                warnings.append(f"Query contains potentially dangerous keyword: {pattern}")

        # Check for SELECT *
        if "SELECT *" in sql_upper:
            warnings.append("Using SELECT * is not recommended, specify columns explicitly")

        # Validate dialect-specific syntax
        if dialect == "influxdb":
            if "JOIN" in sql_upper:
                warnings.append("InfluxDB does not support JOINs in InfluxQL")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )
